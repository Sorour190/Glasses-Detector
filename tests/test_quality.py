import cv2
import numpy as np

from glasses_detector.predict import GlassesDetector, measure_quality


class NoFaceDetector:
    def detect(self, _bgr):
        return None


def test_blur_measurement_does_not_treat_sensor_noise_as_detail():
    rng = np.random.default_rng(4)
    gray = np.clip(90 + rng.normal(0, 20, (160, 160)), 0, 255).astype(np.uint8)
    crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raw_laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    quality = measure_quality(crop)

    assert raw_laplacian > 1_000
    assert quality.blur_score < 2.2


def test_quality_measurement_reports_exposure_and_dynamic_range():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (160, 1))
    crop = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)

    quality = measure_quality(crop)

    assert 126 <= quality.brightness <= 129
    assert 228 <= quality.contrast <= 232


def test_no_face_result_does_not_report_placeholder_quality_as_real_measurements():
    detector = object.__new__(GlassesDetector)
    detector.detector = NoFaceDetector()

    result = detector.predict(np.zeros((80, 80, 3), dtype=np.uint8))

    assert result.face_found is False
    assert result.brightness == 0.0
    assert result.contrast == 0.0
