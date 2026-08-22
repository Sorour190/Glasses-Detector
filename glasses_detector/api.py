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
    GLASSES_CHECKPOINT  path to a .pt checkpoint (default: newest runs/*/best.pt)
    GLASSES_THRESHOLD   fallback decision threshold (default 0.5) — only used when
                        models/threshold.json is absent; the calibrated band wins otherwise
    GLASSES_DEVICE      cuda | cpu (default: cpu, so a training run can keep the GPU)
    GLASSES_AGG_*       multi-frame rule overrides, e.g. GLASSES_AGG_MIN_BLUR=15
                        (see aggregate.AggregateConfig)
    GLASSES_LOG_DIR     if set, every frame (crop + scores + optional ground truth)
                        is appended to <dir>/frames.csv for scripts/review_log.py
    GLASSES_LOG_FULL=1  also keep the full submitted frames

Endpoints:
    POST /validate/glasses        one frame  -> per-frame verdict (kept for old clients)
    POST /validate/glasses/batch  a burst of frames (5 at 5 fps ≈ 1 s) -> per-frame
                                  results + ONE aggregated verdict (stateless).
    POST /validate/glasses/stream one frame at a time with a session_id -> per-frame
                                  result on every call + a verdict on every 5th frame
                                  (tumbling window). This is what the Live tab and a
                                  5 fps production client use.

The full production path runs on every upload: SCRFD face/landmark detection
-> ROI-v1 eye-region crop -> 3-class model -> eyewear = P(eyeglasses)+P(sunglasses).
Sunglasses count as glasses for the verification gate (AggregateConfig.vote_on="eyewear";
set "eyeglasses" for the legacy P(eyeglasses)-only semantics).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .aggregate import AggregateConfig, aggregate, judge_frame
from .framelog import FrameLogger
from .predict import GlassesDetector, GlassesResult

app = FastAPI(title="Glasses Detection", version="2.1.0")

_detector: GlassesDetector | None = None
_agg_cfg: AggregateConfig = AggregateConfig()
_logger: FrameLogger | None = None
MAX_BATCH = 10
TRUTH_VALUES = ("glasses", "none", "unknown")


class ValidationResponse(BaseModel):
    wearing_glasses: bool      # any eyewear (sunglasses included) under the default vote mode
    confidence: float
    probability: float         # P(eyeglasses) — legacy field, telemetry
    eyewear_prob: float = 0.0  # P(eyeglasses) + P(sunglasses): what the decision uses
    uncertain: bool
    face_found: bool
    det_score: float
    class_probs: dict          # {none, eyeglasses, sunglasses} — telemetry
    action: str                # "pass" | "remove_glasses" | "retry_capture"
    blur_score: float = 0.0    # Laplacian variance of the eye crop (higher = sharper)
    eye_dist_px: float = 0.0   # inter-ocular distance in the submitted frame


class BatchFrame(ValidationResponse):
    index: int
    valid: bool                # passed the quality gate and voted
    reject_reason: Optional[str] = None   # no_face | low_det | blurry | too_far
    vote: Optional[str] = None            # glasses | none | unsure


class BatchVerdict(BaseModel):
    action: str                # "pass" | "remove_glasses" | "retry_capture"
    reason: str                # ok | glasses | mixed | no_face | low_det | blurry | too_far
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
    config: dict               # the AggregateConfig used, for telemetry


def _default_checkpoint() -> str:
    if Path("models/glasses_v1.pt").exists():          # shipped release model
        return "models/glasses_v1.pt"
    runs = sorted(Path("runs").glob("*/best.pt"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("no model found; train one or fetch models/glasses_v1.pt")
    return str(runs[-1])


@app.on_event("startup")
def load_model():
    global _detector, _agg_cfg, _logger
    checkpoint = os.environ.get("GLASSES_CHECKPOINT") or _default_checkpoint()
    threshold = float(os.environ.get("GLASSES_THRESHOLD", "0.5"))
    device = os.environ.get("GLASSES_DEVICE", "cpu")
    _detector = GlassesDetector(checkpoint, device=device, threshold=threshold)
    _agg_cfg = AggregateConfig.load()
    log_dir = os.environ.get("GLASSES_LOG_DIR")
    if log_dir:
        _logger = FrameLogger(log_dir, save_full=os.environ.get("GLASSES_LOG_FULL") == "1")
    print(f"loaded {checkpoint} on {device} · aggregate={_agg_cfg.to_dict()}"
          + (f" · logging to {log_dir}" if log_dir else ""))


@app.get("/health")
def health():
    return {"status": "ok"}


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
    """(wearing, uncertain) for one frame under the configured vote mode.
    vote_on="eyewear": sunglasses count as glasses (verification gate)."""
    if not r.face_found:
        return False, True
    if _agg_cfg.vote_on == "eyewear":
        p = r.eyewear_prob
        return p >= _agg_cfg.eyewear_t_high, _agg_cfg.eyewear_t_low <= p < _agg_cfg.eyewear_t_high
    return r.wearing_glasses, r.uncertain


def _frame_action(r: GlassesResult) -> str:
    wearing, uncertain = _frame_decision(r)
    if not r.face_found or uncertain:
        return "retry_capture"
    return "remove_glasses" if wearing else "pass"


def _to_response(r: GlassesResult, action: str) -> dict:
    wearing, uncertain = _frame_decision(r)
    return dict(
        wearing_glasses=wearing,
        confidence=round(r.confidence, 4),
        probability=round(r.probability, 4),
        eyewear_prob=round(r.eyewear_prob, 4),
        uncertain=uncertain,
        face_found=r.face_found,
        det_score=round(r.det_score, 4),
        class_probs=dict(zip(("none", "eyeglasses", "sunglasses"), r.class_probs)),
        action=action,
        blur_score=round(r.blur_score, 2),
        eye_dist_px=round(r.eye_dist_px, 1),
    )


def _check_truth(truth: str) -> str:
    if truth not in TRUTH_VALUES:
        raise HTTPException(status_code=400, detail=f"truth must be one of {TRUTH_VALUES}")
    return truth


@app.post("/validate/glasses", response_model=ValidationResponse)
def validate_glasses(face_image: UploadFile = File(...),
                     truth: str = Form("unknown"), session_id: str = Form("")):
    """Single-frame verdict (legacy / photo upload). Sync def -> runs in the threadpool."""
    truth = _check_truth(truth)
    bgr = _decode(face_image.file.read())
    result = _detector.predict(bgr)
    action = _frame_action(result)
    if _logger:
        _logger.log(result, pred_action=action, truth=truth, session_id=session_id, full_bgr=bgr)
    return ValidationResponse(**_to_response(result, action))


@app.post("/validate/glasses/batch", response_model=BatchResponse)
def validate_glasses_batch(face_images: List[UploadFile] = File(...),
                           truth: str = Form("unknown"), session_id: str = Form("")):
    """Burst verdict: send the 5 frames captured at 5 fps, get ONE decision.

    Frames failing the quality gate (no face, low detector score, blurry, too far)
    do not vote. See aggregate.aggregate() for the rule.
    """
    truth = _check_truth(truth)
    if not 1 <= len(face_images) <= MAX_BATCH:
        raise HTTPException(status_code=400, detail=f"send 1..{MAX_BATCH} frames")
    session_id = session_id or uuid.uuid4().hex[:12]
    t_low, t_high = _band()

    frames_bgr = [_decode(f.file.read()) for f in face_images]
    results = [_detector.predict(bgr) for bgr in frames_bgr]
    verdict = aggregate(results, t_low, t_high, _agg_cfg)

    ts = time.time()
    frames = []
    for r, v, bgr in zip(results, verdict.frames, frames_bgr):
        action = _frame_action(r)
        frames.append(BatchFrame(**_to_response(r, action), index=v.index, valid=v.valid,
                                 reject_reason=v.reject_reason, vote=v.vote))
        if _logger:
            _logger.log(r, pred_action=action, truth=truth, session_id=session_id,
                        frame_idx=v.index, valid=v.valid, reject_reason=v.reject_reason,
                        batch_action=verdict.action, batch_reason=verdict.reason,
                        full_bgr=bgr, ts=ts)

    vd = verdict.to_dict(); vd.pop("frames")
    return BatchResponse(session_id=session_id, verdict=BatchVerdict(**vd),
                         frames=frames, config=_agg_cfg.to_dict())


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
    decision_no: int = 0               # how many decisions this session has produced


class _Session:
    __slots__ = ("results", "frame_no", "decision_no", "last_seen", "lock")

    def __init__(self):
        self.results: list[GlassesResult] = []
        self.frame_no = 0
        self.decision_no = 0
        self.last_seen = time.time()
        self.lock = threading.Lock()


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()
SESSION_TTL_S = 60.0


def _get_session(session_id: str) -> _Session:
    now = time.time()
    with _sessions_lock:
        # cheap GC: drop sessions idle for longer than SESSION_TTL_S
        for sid in [s for s, v in _sessions.items() if now - v.last_seen > SESSION_TTL_S]:
            _sessions.pop(sid, None)
        s = _sessions.get(session_id)
        if s is None:
            s = _sessions[session_id] = _Session()
        s.last_seen = now
        return s


@app.post("/validate/glasses/stream", response_model=StreamResponse)
def validate_glasses_stream(face_image: UploadFile = File(...), session_id: str = Form(...),
                            truth: str = Form("unknown"), reset: bool = Form(False)):
    """Send frames one by one with the same session_id; every n_frames-th frame
    returns a `verdict` for that window (tumbling window, no overlap). Per-frame
    results come back on every call so the UI can show live feedback.
    Pass reset=true on the first frame of a new check to clear the window."""
    truth = _check_truth(truth)
    bgr = _decode(face_image.file.read())
    sess = _get_session(session_id)
    t_low, t_high = _band()
    n = max(1, _agg_cfg.n_frames)

    result = _detector.predict(bgr)
    action = _frame_action(result)
    with sess.lock:
        if reset:
            sess.results.clear(); sess.frame_no = 0; sess.decision_no = 0
        sess.results.append(result)
        sess.frame_no += 1
        frame_no = sess.frame_no
        verdict = None
        if len(sess.results) >= n:
            verdict = aggregate(sess.results, t_low, t_high, _agg_cfg)
            sess.results.clear()
            sess.decision_no += 1
        decision_no = sess.decision_no
        window = len(sess.results)

    # per-frame quality vote (same gate the window uses)
    fv = judge_frame(result, frame_no - 1, t_low, t_high, _agg_cfg)
    frame = BatchFrame(**_to_response(result, action), index=frame_no - 1, valid=fv.valid,
                       reject_reason=fv.reject_reason, vote=fv.vote)
    if _logger:
        _logger.log(result, pred_action=action, truth=truth, session_id=session_id,
                    frame_idx=frame_no - 1, valid=fv.valid, reject_reason=fv.reject_reason,
                    batch_action=verdict.action if verdict else "",
                    batch_reason=verdict.reason if verdict else "", full_bgr=bgr)
    vd = None
    if verdict:
        d = verdict.to_dict(); d.pop("frames"); vd = BatchVerdict(**d)
    return StreamResponse(session_id=session_id, frame_no=frame_no, frame=frame, window=window,
                          decision_every=n, verdict=vd, decision_no=decision_no)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Glasses Check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, sans-serif; background: #10151c; color: #e4e9ef;
         display: flex; flex-direction: column; align-items: center; box-sizing: border-box;
         min-height: 100vh; margin: 0; padding: 1.25rem 1rem 3rem; }
  h1 { font-size: 1.4rem; margin: .25rem 0; } .sub { color: #a8b3c1; margin: 0 0 1rem; text-align: center; }
  nav { display: flex; gap: .4rem; margin-bottom: 1.25rem; }
  nav button { background: #1b2430; color: #c9d3df; border: 1px solid #2c3948; border-radius: 99px;
               padding: .5rem 1rem; font-size: .95rem; }
  nav button.on { background: #2b4a6f; color: #fff; border-color: #5c9bd6; }
  section { display: none; width: 100%; max-width: 520px; flex-direction: column; align-items: center; }
  section.on { display: flex; }
  #drop { border: 2px dashed #3a4a5f; border-radius: 12px; padding: 3rem 2.5rem; width: 100%;
          box-sizing: border-box; text-align: center; cursor: pointer; transition: border-color .15s; }
  #drop:hover, #drop.over { border-color: #5c9bd6; }
  #preview { max-width: 320px; max-height: 320px; border-radius: 8px; margin-top: 1rem; display: none; }
  .verdict { margin-top: 1.25rem; font-size: 1.2rem; font-weight: 600;
             padding: .6rem 1.4rem; border-radius: 99px; display: none; }
  .detail { color: #77828f; font-size: .85rem; margin-top: .6rem; text-align: center; min-height: 1.2em; }
  .stage { position: relative; width: 100%; background: #000; border-radius: 12px; overflow: hidden;
           min-height: 200px; }
  .stage video, .stage canvas { display: block; width: 100%; height: auto; }
  .banner { position: absolute; top: .6rem; left: 50%; transform: translateX(-50%); font-weight: 700;
            font-size: 1.1rem; padding: .4rem 1rem; border-radius: 99px; background: #2a2f36;
            color: #cfd6de; white-space: nowrap; }
  .strip { display: block; width: 100%; height: 14px; margin-top: .5rem; border-radius: 4px; background: #1b2430; }
  .prog { width: 100%; height: 6px; background: #1b2430; border-radius: 3px; margin-top: .6rem; overflow: hidden; }
  .prog > div { height: 100%; width: 0; background: #5c9bd6; }
  .row { display: flex; gap: .5rem; margin-top: .9rem; flex-wrap: wrap; justify-content: center; align-items: center; }
  .act { background: #2b4a6f; color: #fff; border: 0; border-radius: 8px; padding: .6rem 1.1rem; font-size: 1rem; }
  .sec { background: #1b2430; color: #c9d3df; border: 1px solid #2c3948; border-radius: 8px;
         padding: .6rem 1rem; font-size: 1rem; }
  button:disabled { opacity: .45; }
  select { background: #1b2430; color: #e4e9ef; border: 1px solid #2c3948; border-radius: 8px;
           padding: .55rem; font-size: 1rem; }
  .hidden-video { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
  .legend { color: #77828f; font-size: .8rem; margin-top: .4rem; text-align: center; }
  .legend i { display: inline-block; width: .7em; height: .7em; border-radius: 2px; margin: 0 .25em 0 .6em; }
  .seg button { background: #1b2430; color: #c9d3df; border: 1px solid #2c3948; border-radius: 99px;
                padding: .4rem .8rem; font-size: .85rem; }
  .seg button.on { background: #4a3a6f; color: #fff; border-color: #9b7ed6; }
  .yes { background: #16351f; color: #7fd89b; }
  .no  { background: #12263a; color: #7cb5e8; }
  .meh { background: #3a2f14; color: #e0bd6b; }
  .err { background: #3a1414; color: #e08b8b; }
</style></head><body>
<h1>Am I wearing glasses?</h1>
<p class="sub">Any eyewear counts — eyeglasses and sunglasses.</p>
<nav>
  <button id="tab-photo" class="on" onclick="show('photo')">Photo</button>
  <button id="tab-live" onclick="show('live')">Live camera</button>
  <button id="tab-video" onclick="show('video')">Video</button>
</nav>

<section id="photo" class="on">
  <div id="drop">tap to take / pick a photo, or drop an image here
    <input id="file" type="file" accept="image/*" hidden></div>
  <img id="preview">
  <div id="result" class="verdict"></div><div id="detail" class="detail"></div>
</section>

<section id="live">
  <div class="stage">
    <video id="lv" autoplay playsinline muted></video>
    <div id="lv-banner" class="banner">camera off</div>
  </div>
  <canvas id="lv-strip" class="strip"></canvas>
  <div class="legend">last frames: <i style="background:#3ec46d"></i>glasses
    <i style="background:#4a90d9"></i>none <i style="background:#d9a83e"></i>unsure
    <i style="background:#6b7480"></i>rejected (blurry / no face / too far)</div>
  <div id="lv-verdict" class="detail" style="font-size:1rem;color:#c9d3df"></div>
  <div id="lv-stats" class="detail"></div>
  <div class="row">
    <button id="lv-start" class="act" onclick="startLive()">Start check</button>
    <button id="lv-stop" class="sec" onclick="stopLive()" disabled>Stop</button>
    <button id="lv-flip" class="sec" onclick="flipLive()">Flip</button>
  </div>
  <div class="row seg" id="truth">
    <span class="legend" style="margin:0">I am actually:</span>
    <button data-v="glasses" onclick="setTruth('glasses')">wearing glasses</button>
    <button data-v="none" onclick="setTruth('none')">not wearing</button>
    <button data-v="unknown" class="on" onclick="setTruth('unknown')">unknown</button>
  </div>
  <div class="legend">frames stream at 5 fps → one decision every 5 frames · the truth toggle is only logged
    (GLASSES_LOG_DIR) so wrong frames can be reviewed</div>
</section>

<section id="video">
  <div class="row" style="margin-top:0">
    <label class="act" style="cursor:pointer">Choose video<input id="vfile" type="file" accept="video/*" hidden></label>
    <select id="vstep">
      <option value="0.0333">every frame (30/s)</option>
      <option value="0.1">10 frames / s</option>
      <option value="0.2" selected>5 frames / s</option>
      <option value="0.5">2 frames / s</option>
      <option value="1">1 frame / s</option>
    </select>
    <button id="vstart" class="act" onclick="scanVideo()" disabled>Scan</button>
    <button id="vstop" class="sec" onclick="stopScan()" disabled>Stop</button>
  </div>
  <div class="stage" style="margin-top:.9rem">
    <canvas id="vframe"></canvas>
    <div id="v-banner" class="banner">no video loaded</div>
  </div>
  <div class="prog"><div id="vprog"></div></div>
  <canvas id="vtl" class="strip"></canvas>
  <div class="legend">timeline: <i style="background:#3ec46d"></i>glasses <i style="background:#4a90d9"></i>none
    <i style="background:#d9a83e"></i>no face / unsure <i style="background:#d94a4a"></i>error</div>
  <div id="vsum" class="detail"></div>
  <div id="vdet" class="detail"></div>
  <video id="vid" class="hidden-video" playsinline muted preload="auto"></video>
</section>

<script>
const $ = id => document.getElementById(id);
const COLORS = {yes: '#3ec46d', no: '#4a90d9', meh: '#d9a83e', err: '#d94a4a', rej: '#6b7480'};
const BURST_N = 5, BURST_MS = 200;          /* 5 frames at 5 fps, like the production line */
const sleep = ms => new Promise(r => setTimeout(r, ms));

function show(tab) {
  for (const t of ['photo', 'live', 'video']) {
    $(t).classList.toggle('on', t === tab);
    $('tab-' + t).classList.toggle('on', t === tab);
  }
  if (tab !== 'live') stopLive();
  if (tab !== 'video') stopScan();
}
function classify(j) {
  if (!j.face_found) return {cls: 'meh', label: 'no face'};
  if (j.uncertain) return {cls: 'meh', label: 'not sure'};
  if (j.wearing_glasses) return {cls: 'yes', label: 'glasses ✓'};
  return {cls: 'no', label: 'no glasses'};
}
function describe(j) {
  return 'p(eyewear)=' + (j.eyewear_prob !== undefined ? j.eyewear_prob : j.probability).toFixed(3)
    + '  ·  none=' + j.class_probs.none.toFixed(2)
    + '  eyeglasses=' + j.class_probs.eyeglasses.toFixed(2)
    + '  sunglasses=' + j.class_probs.sunglasses.toFixed(2)
    + '  ·  face score=' + j.det_score.toFixed(2)
    + (j.blur_score !== undefined ? '  ·  blur=' + j.blur_score.toFixed(0) : '');
}
async function postBlob(blob) {
  const fd = new FormData(); fd.append('face_image', blob, 'frame.jpg');
  const r = await fetch('/validate/glasses', {method: 'POST', body: fd});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
async function postBatch(blobs, truth, session) {
  const fd = new FormData();
  blobs.forEach((b, i) => fd.append('face_images', b, 'frame' + i + '.jpg'));
  fd.append('truth', truth); fd.append('session_id', session);
  const r = await fetch('/validate/glasses/batch', {method: 'POST', body: fd});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
/* draw the current frame of a <video> onto a canvas, longest side <= maxSide */
function grab(src, cv, maxSide) {
  const w = src.videoWidth, h = src.videoHeight;
  if (!w || !h) return false;
  const s = Math.min(1, (maxSide || 640) / Math.max(w, h));
  cv.width = Math.round(w * s); cv.height = Math.round(h * s);
  cv.getContext('2d').drawImage(src, 0, 0, cv.width, cv.height);
  return true;
}
const toBlob = cv => new Promise(r => cv.toBlob(r, 'image/jpeg', 0.85));
function setBanner(el, cls, text) { el.className = 'banner ' + (cls || ''); el.textContent = text; }
function paintStrip(cv, items, total) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(cv.clientWidth * dpr)); cv.height = Math.round(14 * dpr);
  const ctx = cv.getContext('2d'), w = cv.width / total;
  items.forEach((c, i) => { ctx.fillStyle = COLORS[c]; ctx.fillRect(i * w, 0, Math.ceil(w), cv.height); });
}

/* ---------- photo ---------- */
const drop = $('drop'), inp = $('file'), res = $('result'), det = $('detail'), prev = $('preview');
drop.onclick = () => inp.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over');
                     if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]); };
inp.onchange = () => inp.files[0] && send(inp.files[0]);
async function send(f) {
  prev.src = URL.createObjectURL(f); prev.style.display = 'block';
  res.style.display = 'inline-block'; res.className = 'verdict'; res.textContent = '…';
  det.textContent = '';
  let j;
  try { j = await postBlob(f); }
  catch (e) { res.className = 'verdict err'; res.textContent = 'error'; return; }
  const c = classify(j);
  res.className = 'verdict ' + c.cls; res.textContent = c.label;
  det.textContent = describe(j);
}

/* ---------- live camera: stream frames at 5 fps -> POST /stream -> a decision every 5 frames ---------- */
const lv = $('lv'), lvBanner = $('lv-banner'), lvStats = $('lv-stats'), lvStrip = $('lv-strip');
const lvVerdict = $('lv-verdict');
const work = document.createElement('canvas');
let stream = null, liveToken = 0, facing = 'user', truth = 'unknown';
const liveSession = Math.random().toString(36).slice(2, 10);
function setTruth(v) {
  truth = v;
  document.querySelectorAll('#truth button').forEach(b => b.classList.toggle('on', b.dataset.v === v));
}
const VERDICT = {
  pass:           {cls: 'no',  label: 'NO GLASSES ✓ (pass)'},
  remove_glasses: {cls: 'yes', label: 'GLASSES DETECTED (would stop)'},
  retry_capture:  {cls: 'meh', label: 'UNSURE (retry)'},
};
const HINT = {blurry: 'hold still', no_face: 'face the camera', too_far: 'come closer',
              low_det: 'face the camera', mixed: 'hold still', glasses: '', ok: ''};
function frameCls(f) {
  if (!f.valid) return 'rej';
  return f.vote === 'glasses' ? 'yes' : f.vote === 'none' ? 'no' : 'meh';
}
async function startLive() {
  stopLive();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setBanner(lvBanner, 'err', 'camera blocked: open the https:// address'); return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia(
      {video: {facingMode: facing, width: {ideal: 640}, height: {ideal: 480}}, audio: false});
  } catch (e) { setBanner(lvBanner, 'err', 'camera error: ' + e.message); return; }
  lv.srcObject = stream;
  lv.style.transform = facing === 'user' ? 'scaleX(-1)' : '';
  try { await lv.play(); } catch (e) {}
  $('lv-start').disabled = true; $('lv-stop').disabled = false;
  setBanner(lvBanner, '', 'starting…');
  liveLoop(++liveToken);
}
async function postStream(blob, truth, session, reset) {
  const fd = new FormData(); fd.append('face_image', blob, 'frame.jpg');
  fd.append('truth', truth); fd.append('session_id', session); fd.append('reset', reset ? 'true' : 'false');
  const r = await fetch('/validate/glasses/stream', {method: 'POST', body: fd});
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
/* stream one frame every BURST_MS (5 fps); the server answers every frame and adds a
   verdict on every 5th frame. Frames are shown live; the banner holds the last decision. */
async function liveLoop(tok) {
  const hist = [], counts = {pass: 0, remove_glasses: 0, retry_capture: 0};
  const session = liveSession + '-' + tok;
  let next = performance.now(), first = true, lastVerdict = null, lat = 0;
  while (tok === liveToken && stream) {
    const wait = next - performance.now();
    if (wait > 0) await sleep(wait);
    next = Math.max(next + BURST_MS, performance.now());     /* keep 5 fps cadence, never pile up */
    if (!grab(lv, work)) { await sleep(50); continue; }
    const t0 = performance.now();
    let j;
    try { j = await postStream(await toBlob(work), truth, session, first); first = false; }
    catch (e) { if (tok === liveToken) setBanner(lvBanner, 'err', 'server error'); await sleep(500); continue; }
    if (tok !== liveToken) return;
    lat = performance.now() - t0;
    const f = j.frame;
    hist.push(frameCls(f)); while (hist.length > 40) hist.shift();
    paintStrip(lvStrip, hist, 40);
    if (j.verdict) {
      const v = j.verdict, vv = VERDICT[v.action] || {cls: 'err', label: v.action}, hint = HINT[v.reason] || '';
      lastVerdict = v; counts[v.action] = (counts[v.action] || 0) + 1;
      setBanner(lvBanner, vv.cls, '#' + j.decision_no + ' ' + vv.label + (hint ? ' — ' + hint : ''));
      lvVerdict.textContent = 'decision #' + j.decision_no + ': valid ' + v.n_valid + '/' + v.n_frames
        + ' · votes glasses=' + v.glasses_votes + ' none=' + v.none_votes + ' unsure=' + v.unsure_votes
        + ' · mean p=' + v.mean_p.toFixed(3) + (Object.keys(v.rejected).length ? ' · rejected ' + JSON.stringify(v.rejected) : '')
        + ' · totals: pass ' + counts.pass + ' / stop ' + counts.remove_glasses + ' / retry ' + counts.retry_capture;
    } else if (!lastVerdict) {
      setBanner(lvBanner, '', 'collecting ' + j.window + '/' + j.decision_every + '…');
    }
    const fl = f.valid ? (f.vote === 'glasses' ? 'glasses' : f.vote === 'none' ? 'none' : 'unsure') : 'rejected:' + f.reject_reason;
    lvStats.textContent = 'frame ' + j.frame_no + ' (' + j.window + '/' + j.decision_every + ' to next decision) · ' + fl
      + ' · p(eyewear)=' + f.eyewear_prob.toFixed(2) + ' blur=' + f.blur_score.toFixed(0) + ' det=' + f.det_score.toFixed(2)
      + ' · ' + Math.round(lat) + ' ms/frame';
  }
}
function stopLive() {
  liveToken++;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  lv.srcObject = null;
  $('lv-start').disabled = false; $('lv-stop').disabled = true;
  setBanner(lvBanner, '', 'camera off');
}
function flipLive() { facing = facing === 'user' ? 'environment' : 'user'; if (stream) startLive(); }

/* ---------- video file: seek step by step, POST each frame, paint a timeline ---------- */
const vid = $('vid'), vframe = $('vframe'), vBanner = $('v-banner'), vprog = $('vprog');
const vtl = $('vtl'), vsum = $('vsum'), vdet = $('vdet');
let scanToken = 0;
$('vfile').onchange = () => {
  const f = $('vfile').files[0]; if (!f) return;
  stopScan();
  vid.src = URL.createObjectURL(f); vid.load();
  vsum.textContent = 'loading…'; vdet.textContent = ''; vprog.style.width = '0';
  $('vstart').disabled = true; setBanner(vBanner, '', 'loading…');
};
vid.onloadedmetadata = async () => {
  vsum.textContent = vid.duration.toFixed(1) + ' s · ' + vid.videoWidth + '×' + vid.videoHeight;
  await seekTo(0); grab(vid, vframe);
  $('vstart').disabled = false; setBanner(vBanner, '', 'ready — press Scan');
};
vid.onerror = () => { vsum.textContent = 'this browser could not decode the video';
                      setBanner(vBanner, 'err', 'video error'); };
function seekTo(t) {
  return new Promise(resolve => {
    let done = false;
    const fin = () => { if (done) return; done = true; vid.removeEventListener('seeked', fin); resolve(); };
    vid.addEventListener('seeked', fin);
    setTimeout(fin, 2000);           /* never hang if seeked doesn't fire */
    vid.currentTime = t;
  });
}
async function scanVideo() {
  const tok = ++scanToken;
  $('vstart').disabled = true; $('vstop').disabled = false;
  const step = parseFloat($('vstep').value), D = vid.duration;
  try { await vid.play(); vid.pause(); } catch (e) {}   /* iOS: prime the decoder */
  const counts = {yes: 0, no: 0, meh: 0, err: 0}; let n = 0, summary = '';
  const dpr = window.devicePixelRatio || 1;
  vtl.width = Math.round(vtl.clientWidth * dpr); vtl.height = Math.round(14 * dpr);
  const ctx = vtl.getContext('2d'); ctx.clearRect(0, 0, vtl.width, vtl.height);
  for (let t = 0; t < D && tok === scanToken; t += step) {
    await seekTo(t);
    if (tok !== scanToken) break;
    if (!grab(vid, vframe)) continue;
    let c, j = null;
    try { j = await postBlob(await toBlob(vframe)); c = classify(j); }
    catch (e) { c = {cls: 'err', label: 'server error'}; }
    if (tok !== scanToken) break;
    counts[c.cls]++; n++;
    ctx.fillStyle = COLORS[c.cls];
    ctx.fillRect(t / D * vtl.width, 0, Math.max(1, step / D * vtl.width), vtl.height);
    vprog.style.width = Math.min(100, (t + step) / D * 100) + '%';
    setBanner(vBanner, c.cls, c.label + ' @ ' + t.toFixed(1) + 's');
    summary = n + ' frames · glasses ' + counts.yes + ' (' + Math.round(100 * counts.yes / n) + '%)'
      + ' · no glasses ' + counts.no + ' · no face/unsure ' + counts.meh
      + (counts.err ? ' · errors ' + counts.err : '');
    vsum.textContent = summary;
    if (j) vdet.textContent = describe(j);
  }
  if (tok === scanToken) {
    $('vstart').disabled = false; $('vstop').disabled = true;
    vsum.textContent = 'done · ' + summary;
  }
}
function stopScan() {
  scanToken++;
  $('vstart').disabled = !vid.src; $('vstop').disabled = true;
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE
