import numpy as np
import pandas as pd

from glasses_detector.aggregate import AggregateConfig
from scripts.review_log import (describe, effective_min_blur, gate_impact, pred_label,
                                quality_gate_reasons)


def test_effective_min_blur_uses_profile_config_unless_cli_overrides_it():
    cfg = AggregateConfig(min_blur=3.0, profile_quality={
        "bad_phone": {"min_blur": 1.25},
    })

    assert effective_min_blur("clean", None, cfg) == 3.0
    assert effective_min_blur("bad_phone", None, cfg) == 1.25
    assert effective_min_blur("bad_phone", 4.5, cfg) == 4.5


def test_quality_gate_assigns_one_specific_reason_per_bad_frame():
    frames = pd.DataFrame({
        "blur_score": [1.0, 8.0, 8.0, 8.0, 8.0],
        "brightness": [100.0, 20.0, 240.0, 100.0, 100.0],
        "contrast": [80.0, 80.0, 80.0, 10.0, 80.0],
    })

    reasons = quality_gate_reasons(
        frames, min_blur=3.5, min_brightness=35, max_brightness=220, min_contrast=25
    )

    assert reasons.tolist() == ["blurry", "too_dark", "too_bright", "low_contrast", None]


def test_gate_impact_reports_safety_and_retention_separately():
    labeled = pd.DataFrame({
        "truth": ["glasses", "glasses", "none", "none"],
        "pred": ["none", "none", "none", "none"],
    })
    reasons = pd.Series(["blurry", None, None, "too_dark"], dtype=object)

    impact = gate_impact(labeled, reasons)

    assert impact == {
        "dangerous_misses": 2,
        "dangerous_rejected": 1,
        "dangerous_reject_rate": 0.5,
        "correct_frames": 2,
        "correct_kept": 1,
        "correct_keep_rate": 0.5,
        "all_reject_rate": 0.5,
    }


def test_describe_treats_migrated_missing_metrics_as_no_data():
    assert describe(pd.Series([np.nan, np.nan])) == "n=0"


def test_pred_label_uses_model_probability_when_quality_action_retries():
    model_none_but_rejected = pd.Series({
        "pred_action": "retry_capture", "eyewear": 0.05, "det_score": 0.8,
    })
    model_glasses_but_rejected = pd.Series({
        "pred_action": "retry_capture", "eyewear": 0.8, "det_score": 0.8,
    })

    assert pred_label(model_none_but_rejected) == "none"
    assert pred_label(model_glasses_but_rejected) == "glasses"


def test_pred_label_respects_vote_mode_and_custom_thresholds():
    row = pd.Series({
        "pred_action": "retry_capture", "p": 0.1, "eyewear": 0.4, "det_score": 0.8,
    })

    assert pred_label(row, vote_on="eyeglasses", t_low=0.15, t_high=0.235) == "none"
    assert pred_label(row, vote_on="eyewear", t_low=0.2, t_high=0.5) == "unsure"
    assert pred_label(row, vote_on="eyewear", t_low=0.1, t_high=0.3) == "glasses"
