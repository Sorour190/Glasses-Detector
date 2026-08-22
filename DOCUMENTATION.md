# Glasses Detector — Project Documentation

**Task:** standalone model answering one question — *is this person wearing clear
eyeglasses?* — as a step in an identity-verification pipeline.
**Scope decisions (from the product owner):** binary output; **sunglasses count as
NOT wearing glasses**; face coverings are irrelevant (a masked face is negative
unless eyeglasses are visible); clear/transparent frames and rimless glasses must
be detected.

Final model: **`models/glasses_v1.pt`** (MobileNetV3-Small, 2.5M params) +
**`models/glasses_v1.onnx`** (production export) + **`models/threshold.json`**
(calibrated operating point).

---

## 1. Results

| Benchmark | Accuracy | Precision | Recall | Notes |
|---|---|---|---|---|
| Clean val (13k imgs) | **99.20%** | 98.8% | 98.7% | AUC 0.9995 |
| Worst degraded condition | **95.96%** | — | — | motion blur, max severity |
| SoF natural harsh-light bench (79 unseen subjects) | **96.5%** | 98.3% | 97.9% | started at 60.5% |
| Sunglasses → "glasses" false-positive rate | **0.41%** | — | — | target was ≤2% |
| Calibration (ECE, after temperature scaling) | **0.0055** | — | — | abstain rate 0.23% |

Run history (`runs/runs.csv`):

| Run | Train size | Clean acc | Worst condition | What changed |
|---|---|---|---|---|
| R1 | 2.0k | 94.97% | 87.6% | starter dataset, pipeline wiring |
| R1b | 2.0k | 95.64% | 87.6% | label cleanup round 1 (12 relabels) |
| R2 | 25k | 97.38% | 93.8% | +32-dataset corpus, 3-stage training |
| R2b | 25k | 98.74% | 95.8% | label cleanup round 2 (306 relabels) |
| R3 | 65k | 98.78% | 94.6%* | +MeGlass (47.9k, identity-disjoint) |
| R4 | 66k | 98.82% | 96.1% | harsh-light augs, SoF slice, cleanup round 3 |
| R5 | 87k | **99.20%** | 96.0% | +FFHQ +11.5k pseudo-labeled hard positives |

\* val set grew and got harder at R3; cross-run numbers before/after are not
strictly comparable — which is why frozen benchmarks (SoF) matter.

**The two most important findings:**
1. **Label noise, not model capacity, was the main accuracy ceiling.** Three
   model-assisted cleanup rounds (~600 corrections) produced larger gains than
   any architecture or hyperparameter change.
2. **Synthetic robustness ≠ natural robustness.** The model scored ≥94% on all
   synthetic blur/noise/pose tests while scoring 60.5% on real harsh-light
   photos (SoF). Fixing it required *asymmetric* augmentation (severe
   underexposure, hard shadows) plus a person-disjoint slice of real
   harsh-light data. This mirrors the production failure that motivated the
   original task.

## 2. How it works

Two components: the **face finder** (SCRFD — an off-the-shelf detector, the same
`det_500m.onnx` the company's production pipeline already runs; reused
deliberately for train/serve parity, it knows nothing about glasses) and the
**glasses classifier** (built in this project: data, labels, training,
calibration are all ours; its backbone starts from generic ImageNet-pretrained
weights, standard transfer learning, then all weights are fine-tuned here).

```
photo → SCRFD face detector (det_500m.onnx — reused production component)
      → 5 landmarks (eyes, nose, mouth corners)
      → ROI-v1 crop: de-rotate so the eye line is horizontal, crop 3.0d×2.0d
        around the eyes (d = inter-ocular distance) → 160×160
      → MobileNetV3-Small, 3-class head: none / eyeglasses / sunglasses
      → temperature-scaled softmax → p(eyeglasses)
      → decision: ≥0.235 glasses · <0.15 no glasses · between = retry capture
```

### Multi-frame verdict & quality gate (the production-line decision)

The line captures at 5 fps. One motion-blurred frame can hide a frame/temples
and score "no glasses", so a single frame never decides. The client sends a
burst of 5 frames (≈1 s) to `POST /validate/glasses/batch` and gets ONE verdict
(`glasses_detector/aggregate.py`):

```
per frame   quality gate: face found · det_score ≥ 0.6 · eye distance ≥ 24 px
            · blur (Laplacian variance of the 160px crop) ≥ 30     else REJECTED
            usable frames vote on eyewear = P(eyeglasses)+P(sunglasses):
            eyewear ≥ 0.5 → glasses · < 0.2 → none · else unsure
            (sunglasses STOP the process too — the verification gate needs a bare face;
             `vote_on="eyeglasses"` restores the legacy P(eyeglasses)-only band)
verdict     1. glasses votes ≥ 2                     → remove_glasses   (safety first)
            2. usable frames < 3                     → retry_capture, reason = dominant
                                                       reject cause (blurry / no_face / too_far / low_det)
            3. none votes ≥ ceil(0.8 × usable)       → pass             (4 of 5)
            4. otherwise                             → retry_capture, reason = mixed
```

Knobs (`AggregateConfig`): `n_frames, min_valid, pass_ratio, fail_votes,
min_det_score, min_blur, min_eye_dist, vote_on, eyewear_t_low, eyewear_t_high`. Precedence: defaults ← `"aggregate": {…}`
block in `models/threshold.json` (kept across re-calibration) ← env
`GLASSES_AGG_<KNOB>` (e.g. `GLASSES_AGG_MIN_BLUR=40`). `min_blur=30` is a
conservative starting point — tune it from the line camera with the log below.
(`GLASSES_THRESHOLD` only applies when `threshold.json` is absent; the calibrated
band wins otherwise.)

**Logging & tuning.** Start the server with `GLASSES_LOG_DIR=logs/live`
(`GLASSES_LOG_FULL=1` to also keep full frames): every frame is appended to
`logs/live/frames.csv` (`p`, class probs, `det_score`, `blur_score`, `eye_dist`,
quality verdict, burst verdict, crop path) and its 160×160 crop saved under
`crops/`. The Live tab has an "I am actually: wearing / not wearing / unknown"
toggle that is sent as `truth` and logged, so
`python scripts/review_log.py logs/live` can print truth×prediction tables,
blur distributions of correct vs. wrong frames, a suggested `min_blur`, and
write `review/mismatches.png` — a contact sheet of the wrongly-scored crops.

First live session (49 bursts, glasses on, phone camera, deliberate shaking):
the quality gate rejected 53 frames (36 of which would have voted "none"),
0 false passes once the glasses-on-forehead bursts are excluded, and the
dominant residual error was the model scoring clear/black frames as
*sunglasses* (~40 of 177 valid frames) — which is why the vote moved to
eyewear = 1 − P(none). Blur did not separate right from wrong frames
(`min_blur=30` left as is).

Why a 3-class head for a binary product? Sunglasses share the discriminative
feature (a frame) with eyeglasses and differ on one (lens transmittance).
Giving them their own logit gives the hard-negative boundary its own gradient
path — final sunglasses FPR is 0.41% vs ~2% when folded into "negative".

Why crop from landmarks at train time too? Train/serve mismatch is the classic
silent killer. Every training image passed through the *same* SCRFD + ROI code
path used in production, and training adds Gaussian jitter to the landmarks to
simulate the detector's real localization error.

## 3. Data

~106k images after hygiene, from: glasses-and-coverings (starter),
the mantasu glasses-detector 32-dataset aggregation, MeGlass (47.9k,
identity labels), FFHQ + Zenodo eyeglasses annotations (16.9k eyewear
positives), SoF (33/112 subjects as training slice, 79 reserved as the
untouched natural benchmark), plus ~11.5k eyewear-positives pseudo-labeled by
the R4 model (confident cases auto-labeled; uncertain cases visually reviewed —
they concentrate rimless/clear-frame hard positives).

Hygiene that made the numbers trustworthy:
- **SCRFD prefilter**: every image must produce a face detection (production
  parity; drops product shots and junk).
- **Dedup**: SHA/pHash exact + near-duplicate clustering (banded LSH at 100k
  scale); duplicate images with conflicting labels dropped.
- **Leakage-proof splits**: group-wise assignment over (pHash cluster ∪
  identity), audited to 0 clusters straddling splits every merge.
- **cal split (7.7k)** used only for temperature + thresholds — never for
  model selection.

## 4. Repo guide

| Path | What |
|---|---|
| `glasses_detector/scrfd.py` | standalone SCRFD ONNX inference |
| `glasses_detector/preprocess.py` | ROI-v1 crop + landmark jitter (PREPROCESS_VERSION) |
| `glasses_detector/dataset.py` | manifest-driven dataset, albumentations recipe |
| `glasses_detector/model.py` | MobileNetV3-Small, 3-logit head |
| `glasses_detector/train.py` | 3-stage schedule, bf16, worst-condition selection |
| `glasses_detector/degrade.py` | frozen 6×3 degradation eval suite |
| `glasses_detector/metrics.py` | per-condition eval, ECE, error contact sheets |
| `glasses_detector/calibrate.py` | temperature + t_low/t_high on the cal split |
| `glasses_detector/export_onnx.py` | ONNX export with baked normalization + parity test |
| `glasses_detector/predict.py` / `api.py` | inference class (+ blur / eye-distance quality signals) + web app with single-frame and 5-frame burst endpoints |
| `glasses_detector/aggregate.py` | multi-frame verdict rule + quality gate (`AggregateConfig`) |
| `glasses_detector/framelog.py` | opt-in per-frame CSV + crop logging (`GLASSES_LOG_DIR`) |
| `scripts/` | manifest building, tier ingest, dedup merge, crop cache, pseudo-label, web test, `review_log.py` (tune the gate from logged frames) |
| `tests/test_aggregate.py` | unit tests for the verdict rule |

## 5. Running it

```bash
pip install -r requirements.txt
# CUDA training build: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Web app (open http://127.0.0.1:8000, drop a photo in); phone/live camera: scripts/serve_https.sh
GLASSES_LOG_DIR=logs/live uvicorn glasses_detector.api:app

# Burst verdict, what the production line should call (5 frames at 5 fps)
curl -F face_images=@f0.jpg -F face_images=@f1.jpg -F face_images=@f2.jpg \
     -F face_images=@f3.jpg -F face_images=@f4.jpg http://127.0.0.1:8000/validate/glasses/batch
# -> {"verdict": {"action": "pass|remove_glasses|retry_capture", "reason": ..., ...}, "frames": [...]}

# Review logged frames, tune the blur gate
python scripts/review_log.py logs/live --out review

# Python
from glasses_detector.predict import GlassesDetector
result = GlassesDetector("models/glasses_v1.pt").predict("photo.jpg")

# Spot-check against random web photos
python scripts/web_test.py --checkpoint models/glasses_v1.pt --n 5

# Retrain / continue the ladder
python -m glasses_detector.train --manifest data/manifest_r5.csv \
    --run-name R6 --aug-severity 1.0 --full-epochs 15
```

Training data manifests live outside the repo (`data/`, gitignored) — rebuild
them with `scripts/build_manifest.py` + `scripts/ingest_*.py` from the public
datasets (Kaggle credentials required for some).

## 6. Known limitations & cautions

- **Pince-nez / antique frameless eyewear** is missed (verified on a 1920s
  portrait). Modern rimless glasses ARE detected (verified). Judged
  out-of-scope for webcam verification traffic.
- **Tinted-lens boundary**: photochromic/gradient lenses sit between
  "eyeglasses" and "sunglasses" by design; the abstain band exists partly for
  them. The labeling rule used everywhere: *eyes clearly visible through the
  lens → eyeglasses*.
- **SoF sunglasses labels** are noisy per our rubric (some "sunglasses" rows
  are dark-ish eyeglasses), which inflates the SoF sunglasses-FPR number
  (18.7%) — treat that cell as indicative, not exact.
- **⚠ Licensing (must resolve before production):** MeGlass (MegaFace
  derivative), FFHQ (CC BY-NC-SA), CelebA-adjacent sources and parts of the
  32-dataset aggregation are research/non-commercial. The shipped weights were
  trained on them. Before production use, either obtain legal sign-off or
  retrain on the permissively-licensed subset + in-house captures (the
  manifest carries per-source provenance so this is a filter, not a redo).
- The per-epoch robustness suite (`degrade.py`) is deliberately *disjoint*
  from training augmentation — do not "fix" them to match, that's the point.

## 7. What I'd do next

1. In-house webcam capture set (~50 people, with/without glasses, 3 lightings)
   — the only data that fully closes the domain gap, and the licensing-clean
   positive set.
2. INT8 static quantization (calibrate on `cal` crops; accept only if
   worst-condition drop <0.5pt).
3. A hand-verified frozen `test_clean` for the final report (current numbers
   use the evolving val split; SoF is the only fully frozen benchmark).
4. Production telemetry: log the p(eyeglasses) histogram + abstain rate;
   drift in that histogram is the retraining alarm.

---
*Built 2026-08-17/18 by Claude (Fable 5 + Opus 5 co-planning) with Omar.
Full plan: https://claude.ai/code/artifact/04d7222a-dff6-4be9-a810-80702d42ef8d*
