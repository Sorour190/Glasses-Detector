"""Multi-frame verdict for the verification gate.

The production line captures a short burst (5 frames ≈ 1 s at 5 fps). A single
blurry frame must never produce a false "pass", so each frame is first checked
for quality (face found, detector score, blur, face size) and only the usable
frames vote. The rule is asymmetric on purpose: glasses evidence wins first.

Pure functions, no I/O — see tests/test_aggregate.py.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional

from .predict import GlassesResult

REJECT_REASONS = ("no_face", "low_det", "blurry", "too_far", "too_dark",
                  "too_bright", "low_contrast")


@dataclass
class AggregateConfig:
    n_frames: int = 5            # expected burst length (1 s at 5 fps); informational
    min_valid: int = 3           # fewer usable frames -> retry_capture
    pass_ratio: float = 0.8      # share of usable frames that must vote "none" (4 of 5)
    fail_votes: int = 2          # >= this many "glasses" frames -> remove_glasses
    min_det_score: float = 0.6   # SCRFD score below this -> frame rejected (low_det)
    min_blur: float = 2.2        # quality-v2: denoised Laplacian variance on 160px crop
                                 # (keeps about 95% of correct frames in current live logs)
    min_eye_dist: float = 24.0   # px between the eyes in the submitted frame (too_far)
    min_brightness: float = 35.0 # mean gray below this -> too_dark
    max_brightness: float = 220.0 # mean gray above this -> too_bright
    min_contrast: float = 25.0   # grayscale p95-p5 below this -> low_contrast
    # What a valid frame votes on. "eyewear" = P(eyeglasses)+P(sunglasses) — sunglasses stop
    # the process too (verification gate). "eyeglasses" = legacy P(eyeglasses) with the
    # calibrated t_low/t_high band from threshold.json.
    vote_on: str = "eyewear"
    eyewear_t_low: float = 0.2   # eyewear below this -> "none" vote
    eyewear_t_high: float = 0.5  # eyewear at/above this -> "glasses" vote; between -> unsure
    profile_quality: dict = field(default_factory=dict)
    # Per bad-camera profile overrides of the vote band, e.g.
    # {"extreme": {"eyewear_t_low": 0.03, "eyewear_t_high": 0.3}}. The model's
    # eyewear probability on true glasses-wearers drops with degradation, so a
    # single band cannot serve all profiles: a low t_low turns collapsed
    # low-confidence frames into "unsure" votes (-> retry) instead of false
    # "none" votes (-> false pass). Fitted by scripts/fit_profile_thresholds.py.
    profile_thresholds: dict = field(default_factory=dict)
    # Profiles whose ROI crop is restoration-enhanced (restore.restore_crop)
    # before the model sees it. Quality gates still use the raw crop.
    restore_profiles: list = field(default_factory=lambda: ["bad_phone", "extreme"])

    @classmethod
    def load(cls, threshold_file: Optional[str] = "models/threshold.json",
             env: Optional[dict] = None) -> "AggregateConfig":
        """defaults <- "aggregate" block of threshold.json <- GLASSES_AGG_* env vars."""
        cfg = cls()
        if threshold_file and os.path.exists(threshold_file):
            import json
            with open(threshold_file) as f:
                block = json.load(f).get("aggregate") or {}
            cfg.update(block)
        env = os.environ if env is None else env
        prefix = "GLASSES_AGG_"
        cfg.update({k[len(prefix):].lower(): v for k, v in env.items() if k.startswith(prefix)})
        return cfg

    def update(self, values: dict) -> None:
        for k, v in values.items():
            if not hasattr(self, k):
                continue
            cur = getattr(self, k)
            if isinstance(cur, (dict, list)) and isinstance(v, str):
                # env vars arrive as strings; container fields take JSON,
                # e.g. GLASSES_AGG_PROFILE_THRESHOLDS='{"extreme": {...}}'
                import json
                v = json.loads(v)
            setattr(self, k, type(cur)(v))

    def to_dict(self) -> dict:
        return asdict(self)

    def for_profile(self, profile: str) -> "AggregateConfig":
        cfg = replace(
            self,
            profile_quality={name: dict(values)
                             for name, values in self.profile_quality.items()},
            profile_thresholds={name: dict(values)
                                for name, values in self.profile_thresholds.items()},
            restore_profiles=list(self.restore_profiles),
        )
        cfg.update(self.profile_quality.get(profile, {}))
        cfg.update(self.profile_thresholds.get(profile, {}))
        return cfg

    def should_restore(self, profile: str) -> bool:
        return profile in self.restore_profiles


@dataclass
class FrameVote:
    index: int
    valid: bool
    reject_reason: Optional[str]     # one of REJECT_REASONS or None
    vote: Optional[str]              # "glasses" | "none" | "unsure" | None (if not valid)


@dataclass
class AggregateVerdict:
    action: str                      # "pass" | "remove_glasses" | "retry_capture"
    reason: str                      # "ok" | "glasses" | "mixed" | a REJECT_REASON
    n_frames: int
    n_valid: int
    glasses_votes: int
    none_votes: int
    unsure_votes: int
    mean_p: float                    # mean p(eyeglasses) over valid frames (telemetry)
    max_p: float
    rejected: dict = field(default_factory=dict)   # {reason: count}
    frames: list = field(default_factory=list)     # list[FrameVote]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean_p"] = round(self.mean_p, 4)
        d["max_p"] = round(self.max_p, 4)
        return d


def judge_frame(r: GlassesResult, index: int, t_low: float, t_high: float,
                cfg: AggregateConfig) -> FrameVote:
    if not r.face_found:
        return FrameVote(index, False, "no_face", None)
    if r.det_score < cfg.min_det_score:
        return FrameVote(index, False, "low_det", None)
    if r.eye_dist_px < cfg.min_eye_dist:
        return FrameVote(index, False, "too_far", None)
    if r.blur_score < cfg.min_blur:
        return FrameVote(index, False, "blurry", None)
    if r.brightness < cfg.min_brightness:
        return FrameVote(index, False, "too_dark", None)
    if r.brightness > cfg.max_brightness:
        return FrameVote(index, False, "too_bright", None)
    if r.contrast < cfg.min_contrast:
        return FrameVote(index, False, "low_contrast", None)
    p, lo, hi = vote_value(r, t_low, t_high, cfg)
    vote = "glasses" if p >= hi else ("none" if p < lo else "unsure")
    return FrameVote(index, True, None, vote)


def vote_value(r: GlassesResult, t_low: float, t_high: float,
               cfg: AggregateConfig) -> tuple[float, float, float]:
    """(probability the frame votes on, t_low, t_high) for the configured mode."""
    if cfg.vote_on == "eyewear":
        return r.eyewear_prob, cfg.eyewear_t_low, cfg.eyewear_t_high
    return r.probability, t_low, t_high


def aggregate(results: Iterable[GlassesResult], t_low: float, t_high: float,
              cfg: Optional[AggregateConfig] = None) -> AggregateVerdict:
    cfg = cfg or AggregateConfig()
    results = list(results)
    votes = [judge_frame(r, i, t_low, t_high, cfg) for i, r in enumerate(results)]
    valid = [(r, v) for r, v in zip(results, votes) if v.valid]
    counts = Counter(v.vote for _, v in valid)
    rejected = Counter(v.reject_reason for v in votes if not v.valid)
    n_valid = len(valid)
    g, n, u = counts["glasses"], counts["none"], counts["unsure"]
    ps = [vote_value(r, t_low, t_high, cfg)[0] for r, _ in valid]
    mean_p = sum(ps) / len(ps) if ps else 0.0
    max_p = max(ps) if ps else 0.0

    # 1. safety direction first: enough glasses evidence -> stop, even if blurry otherwise
    if g >= cfg.fail_votes:
        action, reason = "remove_glasses", "glasses"
    # 2. not enough usable frames -> ask to retry, naming the dominant cause
    elif n_valid < cfg.min_valid:
        action = "retry_capture"
        reason = rejected.most_common(1)[0][0] if rejected else "mixed"
    # 3. clear majority of clean frames -> pass
    elif n >= math.ceil(cfg.pass_ratio * n_valid):
        action, reason = "pass", "ok"
    else:
        action, reason = "retry_capture", "mixed"

    return AggregateVerdict(action, reason, len(results), n_valid, g, n, u,
                            mean_p, max_p, dict(rejected), votes)


def combine_profile_verdicts(verdicts: dict[str, AggregateVerdict]) -> AggregateVerdict:
    usable = [v for v in verdicts.values() if v.action != "retry_capture"]
    glasses = [v for v in verdicts.values() if v.action == "remove_glasses"]
    passes = [name for name, v in verdicts.items() if v.action == "pass"]
    retries = [v for v in verdicts.values() if v.action == "retry_capture"]
    degraded_passes = [name for name in passes if name != "clean"]

    if glasses:
        action, reason = "remove_glasses", "glasses"
    elif "clean" in passes and len(degraded_passes) >= 2:
        action, reason = "pass", "ok"
    else:
        action, reason = "retry_capture", "profile_consensus"

    ps = [v.mean_p for v in usable]
    rejected = Counter(v.reason for v in retries)
    return AggregateVerdict(
        action=action,
        reason=reason,
        n_frames=len(verdicts),
        n_valid=len(usable),
        glasses_votes=len(glasses),
        none_votes=len(passes),
        unsure_votes=len(retries),
        mean_p=sum(ps) / len(ps) if ps else 0.0,
        max_p=max((v.max_p for v in usable), default=0.0),
        rejected=dict(rejected),
    )
