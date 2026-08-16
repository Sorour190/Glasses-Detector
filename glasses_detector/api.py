"""FastAPI validation endpoint for the onboarding flow.

Run:
    GLASSES_CHECKPOINT=checkpoints/glasses.pt uvicorn glasses_detector.api:app

The endpoint receives the face crop already extracted by the identity step
and returns whether the user is wearing glasses, so the onboarding flow can
ask them to remove them and retry.
"""

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .predict import GlassesDetector

app = FastAPI(title="Glasses Detection", version="1.0.0")

_detector: GlassesDetector = None


class ValidationResponse(BaseModel):
    wearing_glasses: bool
    confidence: float
    probability: float
    uncertain: bool
    # What the onboarding flow should do: "pass", "remove_glasses", or "retry_capture"
    action: str


@app.on_event("startup")
def load_model():
    global _detector
    checkpoint = os.environ.get("GLASSES_CHECKPOINT", "checkpoints/glasses.pt")
    _detector = GlassesDetector(checkpoint)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate/glasses", response_model=ValidationResponse)
async def validate_glasses(face_image: UploadFile = File(...)):
    data = await face_image.read()
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    result = _detector.predict(image)

    if result.uncertain:
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
        action=action,
    )
