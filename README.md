# Glasses Detector — Onboarding Validation Step

Binary classifier that checks whether the person in a face crop is wearing
glasses. It slots in next to the existing identity-verification model in the
onboarding flow: the identity step already localizes the face, so this model
takes the same face crop and returns wearing glasses / not wearing glasses.

## How it works

- **Model:** MobileNetV3-Small (ImageNet pretrained) with a single-logit head —
  small (~2.5M params), fast on CPU, ideal for a synchronous validation step.
- **Training:** two-stage transfer learning — warm up the new head with the
  backbone frozen, then fine-tune the last backbone blocks at a lower LR.
- **Output:** `P(wearing glasses)` plus an *uncertainty band*: predictions near
  the threshold return `action: retry_capture` instead of a hard accept/reject,
  which avoids wrongly bouncing users on blurry or badly lit frames.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get training data

The quickest start is CelebA, which labels ~200k face images with an
`Eyeglasses` attribute:

```bash
python scripts/prepare_celeba.py \
    --images img_align_celeba \
    --attrs list_attr_celeba.txt \
    --out data
```

This produces the folder layout the trainer expects (and balances the classes,
since only ~6% of CelebA has glasses):

```
data/
  train/glasses/  train/no_glasses/
  val/glasses/    val/no_glasses/
```

Any dataset in this layout works. **For best production accuracy, add face
crops from your own capture pipeline** — same camera framing and lighting the
onboarding flow actually produces.

## 2. Train

```bash
python -m glasses_detector.train --data-dir data --out checkpoints/glasses.pt
```

Trains in minutes on a GPU, and is feasible on CPU with a subset. Expect
~98%+ validation accuracy on CelebA.

## 3. Use it

### In Python

```python
from glasses_detector import GlassesDetector

detector = GlassesDetector("checkpoints/glasses.pt")
result = detector.predict("face_crop.jpg")  # path, PIL image, or numpy array

result.wearing_glasses  # bool
result.probability      # raw P(glasses)
result.uncertain        # True near the threshold -> ask user to retry capture
```

### As a service

```bash
GLASSES_CHECKPOINT=checkpoints/glasses.pt uvicorn glasses_detector.api:app --port 8000
```

```bash
curl -F "face_image=@face_crop.jpg" http://localhost:8000/validate/glasses
```

```json
{
  "wearing_glasses": true,
  "confidence": 0.97,
  "probability": 0.97,
  "uncertain": false,
  "action": "remove_glasses"
}
```

`action` is what the onboarding flow should do: `pass`, `remove_glasses`
(prompt the user to take them off and re-capture), or `retry_capture`
(low-confidence frame — capture again before deciding).

### Production inference (ONNX)

```bash
python -m glasses_detector.export_onnx --checkpoint checkpoints/glasses.pt --out glasses.onnx
```

Run with `onnxruntime` (no PyTorch dependency): resize the face crop to
224×224 RGB, scale to [0,1], normalize with ImageNet mean/std, and apply
`sigmoid` to the output logit.

## Tests

```bash
pytest tests/
```

## Notes & tuning

- **Threshold:** default 0.5. If missing glasses is costlier than a false
  rejection (usual case for onboarding compliance), lower the threshold
  (e.g. 0.35) so borderline cases fail toward "glasses detected".
- **Uncertainty band:** default ±0.15 around the threshold; widen it to be
  stricter about frame quality.
- **Sunglasses vs. clear glasses:** CelebA's attribute covers both. If you need
  to distinguish them (e.g. sunglasses always rejected, clear glasses allowed),
  retrain with three classes — the pipeline structure stays the same.
