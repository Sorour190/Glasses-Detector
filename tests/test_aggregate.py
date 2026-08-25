"""Unit tests for the multi-frame verdict (pure function, no model needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glasses_detector.aggregate import (                            # noqa: E402
    AggregateConfig,
    AggregateVerdict,
    aggregate,
    combine_profile_verdicts,
    judge_frame,
)
from glasses_detector.predict import GlassesResult                  # noqa: E402

T_LOW, T_HIGH = 0.15, 0.235
CFG = AggregateConfig()   # n=5, min_valid=3, pass_ratio=0.8, fail_votes=2, min_blur=2.2,
                          # vote_on=eyewear with eyewear_t_low=0.2 / t_high=0.5


def frame(p=0.05, face=True, det=0.9, blur=100.0, eye=60.0, sun=0.0,
          brightness=128.0, contrast=100.0):
    """p = P(eyeglasses), sun = P(sunglasses); eyewear = p + sun."""
    wearing = p >= T_HIGH
    return GlassesResult(wearing, max(p, 1 - p), p, T_LOW <= p < T_HIGH, face, det,
                         (1 - p - sun, p, sun), eyewear_prob=p + sun,
                         blur_score=blur, eye_dist_px=eye,
                         brightness=brightness, contrast=contrast)


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
    v = aggregate([frame(p=0.02, blur=0.5), frame(p=0.03, blur=1), frame(p=0.01, blur=2),
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


def test_bad_exposure_and_low_contrast_are_rejected_with_specific_reasons():
    v = aggregate([
        frame(brightness=20),
        frame(brightness=240),
        frame(contrast=5),
        frame(),
        frame(),
    ], T_LOW, T_HIGH, CFG)

    assert v.action == "retry_capture" and v.n_valid == 2
    assert v.rejected == {"too_dark": 1, "too_bright": 1, "low_contrast": 1}


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


def test_shipped_quality_v2_gate_matches_the_runtime_defaults():
    threshold_file = Path(__file__).resolve().parents[1] / "models" / "threshold.json"
    shipped = AggregateConfig.load(str(threshold_file), env={})
    defaults = AggregateConfig()

    assert (shipped.min_blur, shipped.min_brightness, shipped.max_brightness,
            shipped.min_contrast) == (
                defaults.min_blur, defaults.min_brightness, defaults.max_brightness,
                defaults.min_contrast,
            )
    assert shipped.for_profile("clean").min_blur == 2.2
    assert shipped.for_profile("mild").min_blur == 2.1
    assert shipped.for_profile("bad_phone").min_blur == 1.9
    assert shipped.for_profile("extreme").min_blur == 1.7


def test_frames_in_verdict_keep_order_and_reasons():
    v = aggregate([frame(face=False), frame(p=0.9), frame(blur=1)], T_LOW, T_HIGH, CFG)
    assert [fv.index for fv in v.frames] == [0, 1, 2]
    assert [fv.reject_reason for fv in v.frames] == ["no_face", None, "blurry"]
    assert [fv.vote for fv in v.frames] == [None, "glasses", None]


def test_bad_phone_uses_a_lower_blur_gate_than_clean():
    cfg = AggregateConfig(profile_quality={"bad_phone": {"min_blur": 1.9}})
    result = frame(blur=2.0)

    clean = judge_frame(result, 0, T_LOW, T_HIGH, cfg.for_profile("clean"))
    bad_phone = judge_frame(result, 0, T_LOW, T_HIGH, cfg.for_profile("bad_phone"))

    assert clean.valid is False and clean.reject_reason == "blurry"
    assert bad_phone.valid is True and bad_phone.vote == "none"


def _profile_verdict(action: str, reason: str = "ok") -> AggregateVerdict:
    return AggregateVerdict(
        action=action,
        reason=reason,
        n_frames=5,
        n_valid=5 if action != "retry_capture" else 0,
        glasses_votes=5 if action == "remove_glasses" else 0,
        none_votes=5 if action == "pass" else 0,
        unsure_votes=0,
        mean_p=0.9 if action == "remove_glasses" else 0.05,
        max_p=0.95 if action == "remove_glasses" else 0.08,
    )


def test_combined_profiles_pass_with_clean_and_two_degraded_passes():
    combined = combine_profile_verdicts({
        "clean": _profile_verdict("pass"),
        "mild": _profile_verdict("pass"),
        "bad_phone": _profile_verdict("pass"),
        "extreme": _profile_verdict("retry_capture", "blurry"),
    })

    assert combined.action == "pass"
    assert combined.none_votes == 3
    assert combined.n_valid == 3


def test_any_usable_profile_detecting_glasses_stops_combined_decision():
    combined = combine_profile_verdicts({
        "clean": _profile_verdict("pass"),
        "mild": _profile_verdict("remove_glasses", "glasses"),
        "bad_phone": _profile_verdict("pass"),
        "extreme": _profile_verdict("retry_capture", "blurry"),
    })

    assert combined.action == "remove_glasses"
    assert combined.glasses_votes == 1


def test_combined_profiles_retry_without_clean_and_two_degraded_passes():
    combined = combine_profile_verdicts({
        "clean": _profile_verdict("pass"),
        "mild": _profile_verdict("pass"),
        "bad_phone": _profile_verdict("retry_capture", "blurry"),
        "extreme": _profile_verdict("retry_capture", "too_dark"),
    })

    assert combined.action == "retry_capture"
    assert combined.reason == "profile_consensus"
