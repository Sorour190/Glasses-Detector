"""Unit tests for the multi-frame verdict (pure function, no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glasses_detector.aggregate import AggregateConfig, aggregate   # noqa: E402
from glasses_detector.predict import GlassesResult                  # noqa: E402

T_LOW, T_HIGH = 0.15, 0.235
CFG = AggregateConfig()   # n=5, min_valid=3, pass_ratio=0.8, fail_votes=2, min_blur=30,
                          # vote_on=eyewear with eyewear_t_low=0.2 / t_high=0.5


def frame(p=0.05, face=True, det=0.9, blur=100.0, eye=60.0, sun=0.0):
    """p = P(eyeglasses), sun = P(sunglasses); eyewear = p + sun."""
    wearing = p >= T_HIGH
    return GlassesResult(wearing, max(p, 1 - p), p, T_LOW <= p < T_HIGH, face, det,
                         (1 - p - sun, p, sun), eyewear_prob=p + sun,
                         blur_score=blur, eye_dist_px=eye)


def test_all_clean_passes():
    v = aggregate([frame() for _ in range(5)], T_LOW, T_HIGH, CFG)
    assert v.action == "pass" and v.reason == "ok"
    assert v.n_valid == 5 and v.none_votes == 5


def test_four_of_five_none_with_one_stray_glasses_passes():
    v = aggregate([frame(), frame(), frame(), frame(), frame(p=0.6)], T_LOW, T_HIGH, CFG)
    assert v.action == "pass"
    assert v.glasses_votes == 1


def test_two_glasses_votes_stop_even_if_others_clean():
    v = aggregate([frame(), frame(), frame(), frame(p=0.5), frame(p=0.7)], T_LOW, T_HIGH, CFG)
    assert v.action == "remove_glasses" and v.reason == "glasses"


def test_glasses_wins_over_too_few_valid_frames():
    # 2 sharp glasses frames + 3 blurry: stop, don't ask to retry
    v = aggregate([frame(p=0.9), frame(p=0.8), frame(blur=3), frame(blur=3), frame(blur=3)],
                  T_LOW, T_HIGH, CFG)
    assert v.action == "remove_glasses"


def test_blurry_burst_is_retry_not_pass():
    # the reported failure mode: shaking camera -> blurry frames scored as "none"
    v = aggregate([frame(p=0.02, blur=5), frame(p=0.03, blur=8), frame(p=0.01, blur=2),
                   frame(p=0.02, blur=50), frame(p=0.7, blur=90)], T_LOW, T_HIGH, CFG)
    assert v.action == "retry_capture" and v.reason == "blurry"
    assert v.n_valid == 2 and v.rejected == {"blurry": 3}


def test_no_face_retry_with_reason():
    v = aggregate([frame(face=False)] * 5, T_LOW, T_HIGH, CFG)
    assert v.action == "retry_capture" and v.reason == "no_face" and v.n_valid == 0
    assert v.mean_p == 0.0


def test_too_far_and_low_det_are_rejected():
    v = aggregate([frame(eye=10), frame(det=0.3), frame(), frame(), frame()], T_LOW, T_HIGH, CFG)
    assert v.action == "pass" and v.n_valid == 3
    assert v.rejected == {"too_far": 1, "low_det": 1}


def test_mixed_unsure_is_retry():
    v = aggregate([frame(), frame(), frame(p=0.3), frame(p=0.3), frame(p=0.3)], T_LOW, T_HIGH, CFG)
    assert v.action == "retry_capture" and v.reason == "mixed"
    assert v.unsure_votes == 3


def test_short_burst_three_clean_frames_passes():
    v = aggregate([frame(), frame(), frame()], T_LOW, T_HIGH, CFG)
    assert v.action == "pass"


def test_pass_ratio_ceil():
    # 4 valid frames, ratio .8 -> need ceil(3.2)=4 none votes; 3 is not enough
    v = aggregate([frame(), frame(), frame(), frame(p=0.3)], T_LOW, T_HIGH, CFG)
    assert v.action == "retry_capture" and v.reason == "mixed"


def test_sunglasses_count_as_glasses_by_default():
    # the live-camera finding: clear/black frames scored as "sunglasses" must still stop
    v = aggregate([frame(p=0.05, sun=0.9)] * 5, T_LOW, T_HIGH, CFG)
    assert v.action == "remove_glasses" and v.glasses_votes == 5


def test_legacy_eyeglasses_mode_ignores_sunglasses():
    cfg = AggregateConfig(vote_on="eyeglasses")
    v = aggregate([frame(p=0.05, sun=0.9)] * 5, T_LOW, T_HIGH, cfg)
    assert v.action == "pass" and v.none_votes == 5


def test_config_env_override_and_json_block(tmp_path, monkeypatch):
    f = tmp_path / "threshold.json"
    f.write_text('{"T": 1.0, "t_low": 0.1, "t_high": 0.3, "aggregate": {"min_blur": 5, "fail_votes": 3}}')
    cfg = AggregateConfig.load(str(f), env={"GLASSES_AGG_MIN_BLUR": "7.5", "GLASSES_AGG_BOGUS": "1"})
    assert cfg.min_blur == 7.5 and cfg.fail_votes == 3 and cfg.pass_ratio == 0.8
    assert not hasattr(cfg, "bogus")


def test_frames_in_verdict_keep_order_and_reasons():
    v = aggregate([frame(face=False), frame(p=0.9), frame(blur=1)], T_LOW, T_HIGH, CFG)
    assert [fv.index for fv in v.frames] == [0, 1, 2]
    assert [fv.reject_reason for fv in v.frames] == ["no_face", None, "blurry"]
    assert [fv.vote for fv in v.frames] == [None, "glasses", None]
