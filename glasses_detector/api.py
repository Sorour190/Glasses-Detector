"""FastAPI app: upload a photo, get a glasses / no-glasses verdict.

Run from the repo root:
    uvicorn glasses_detector.api:app --host 127.0.0.1 --port 8000
then open http://127.0.0.1:8000 in a browser and drop a photo in.

Env overrides:
    GLASSES_CHECKPOINT  path to a .pt checkpoint (default: newest runs/*/best.pt)
    GLASSES_THRESHOLD   decision threshold (default 0.5)
    GLASSES_DEVICE      cuda | cpu (default: cpu, so a training run can keep the GPU)

The full production path runs on every upload: SCRFD face/landmark detection
-> ROI-v1 eye-region crop -> 3-class model -> p(eyeglasses). Sunglasses count
as NOT wearing glasses by design.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .predict import GlassesDetector

app = FastAPI(title="Glasses Detection", version="2.0.0")

_detector: GlassesDetector | None = None


class ValidationResponse(BaseModel):
    wearing_glasses: bool
    confidence: float
    probability: float
    uncertain: bool
    face_found: bool
    det_score: float
    class_probs: dict          # {none, eyeglasses, sunglasses} — telemetry
    action: str                # "pass" | "remove_glasses" | "retry_capture"


def _default_checkpoint() -> str:
    runs = sorted(Path("runs").glob("*/best.pt"), key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError("no runs/*/best.pt found; train a model first")
    return str(runs[-1])


@app.on_event("startup")
def load_model():
    global _detector
    checkpoint = os.environ.get("GLASSES_CHECKPOINT") or _default_checkpoint()
    threshold = float(os.environ.get("GLASSES_THRESHOLD", "0.5"))
    device = os.environ.get("GLASSES_DEVICE", "cpu")
    _detector = GlassesDetector(checkpoint, device=device, threshold=threshold)
    print(f"loaded {checkpoint} on {device}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate/glasses", response_model=ValidationResponse)
async def validate_glasses(face_image: UploadFile = File(...)):
    data = await face_image.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    result = _detector.predict(bgr)

    if not result.face_found or result.uncertain:
        action = "retry_capture"
    elif result.wearing_glasses:
        action = "remove_glasses"
    else:
        action = "pass"

    return ValidationResponse(
        wearing_glasses=result.wearing_glasses,
        confidence=round(result.confidence, 4),
        probability=round(result.probability, 4),
        uncertain=result.uncertain,
        face_found=result.face_found,
        det_score=round(result.det_score, 4),
        class_probs=dict(zip(("none", "eyeglasses", "sunglasses"),
                             result.class_probs)),
        action=action,
    )


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Glasses Check</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: system-ui, sans-serif; background: #10151c; color: #e4e9ef;
         display: flex; flex-direction: column; align-items: center;
         min-height: 100vh; margin: 0; padding: 2rem 1rem; }
  h1 { font-size: 1.4rem; } .sub { color: #a8b3c1; margin: 0 0 1.5rem; }
  #drop { border: 2px dashed #3a4a5f; border-radius: 12px; padding: 3rem 2.5rem;
          text-align: center; cursor: pointer; transition: border-color .15s; }
  #drop:hover, #drop.over { border-color: #5c9bd6; }
  #preview { max-width: 320px; max-height: 320px; border-radius: 8px;
             margin-top: 1rem; display: none; }
  #result { margin-top: 1.25rem; font-size: 1.2rem; font-weight: 600;
            padding: .6rem 1.4rem; border-radius: 99px; display: none; }
  .yes { background: #16351f; color: #7fd89b; }
  .no  { background: #12263a; color: #7cb5e8; }
  .meh { background: #3a2f14; color: #e0bd6b; }
  #detail { color: #77828f; font-size: .85rem; margin-top: .6rem; }
</style></head><body>
<h1>Am I wearing glasses?</h1>
<p class="sub">Drop a photo — clear eyeglasses count, sunglasses don't.</p>
<div id="drop">click or drop an image here
  <input id="file" type="file" accept="image/*" hidden></div>
<img id="preview">
<div id="result"></div><div id="detail"></div>
<script>
const drop = document.getElementById('drop'), inp = document.getElementById('file');
const res = document.getElementById('result'), det = document.getElementById('detail');
const prev = document.getElementById('preview');
drop.onclick = () => inp.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over');
                     if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]); };
inp.onchange = () => inp.files[0] && send(inp.files[0]);
async function send(f) {
  prev.src = URL.createObjectURL(f); prev.style.display = 'block';
  res.style.display = 'inline-block'; res.className = ''; res.textContent = '…';
  det.textContent = '';
  const fd = new FormData(); fd.append('face_image', f);
  const r = await fetch('/validate/glasses', {method: 'POST', body: fd});
  if (!r.ok) { res.textContent = 'error'; return; }
  const j = await r.json();
  if (!j.face_found) { res.className = 'meh'; res.textContent = 'no face found'; }
  else if (j.uncertain) { res.className = 'meh';
    res.textContent = 'not sure — try another photo'; }
  else if (j.wearing_glasses) { res.className = 'yes'; res.textContent = 'glasses ✓'; }
  else { res.className = 'no'; res.textContent = 'no glasses'; }
  det.textContent = 'p(eyeglasses)=' + j.probability.toFixed(3)
    + '  ·  none=' + j.class_probs.none.toFixed(2)
    + '  eyeglasses=' + j.class_probs.eyeglasses.toFixed(2)
    + '  sunglasses=' + j.class_probs.sunglasses.toFixed(2)
    + '  ·  face score=' + j.det_score.toFixed(2);
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE
