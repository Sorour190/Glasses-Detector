import cv2
import numpy as np
import pytest

from glasses_detector.degrade import apply_bad_camera


def _checkerboard(size: int = 192) -> np.ndarray:
    yy, xx = np.indices((size, size))
    gray = (((xx // 4 + yy // 4) % 2) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_clean_profile_preserves_the_submitted_frame():
    frame = _checkerboard()

    actual = apply_bad_camera(frame, "clean", index=7)

    assert np.array_equal(actual, frame)
    assert actual is not frame


def test_bad_phone_profile_is_deterministic_and_keeps_frame_contract():
    frame = _checkerboard()

    first = apply_bad_camera(frame, "bad_phone", index=11)
    second = apply_bad_camera(frame, "bad_phone", index=11)

    assert np.array_equal(first, second)
    assert first.shape == frame.shape
    assert first.dtype == np.uint8
    assert not np.array_equal(first, frame)


def test_extreme_profile_reduces_full_frame_detail():
    frame = _checkerboard()

    degraded = apply_bad_camera(frame, "extreme", index=3)

    assert _sharpness(degraded) < _sharpness(frame) * 0.1


def test_extreme_profile_is_dark_but_still_human_usable():
    frame = np.full((192, 192, 3), 128, dtype=np.uint8)

    degraded = apply_bad_camera(frame, "extreme", index=3)

    assert 55.0 <= float(degraded.mean()) <= 75.0


def test_unknown_bad_camera_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown bad-camera profile"):
        apply_bad_camera(_checkerboard(), "potato")
