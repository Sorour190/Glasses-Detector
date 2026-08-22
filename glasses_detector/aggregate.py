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
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from .predict import GlassesResult

REJECT_REASONS = ("no_face", "low_det", "blurry", "too_far")


@dataclass
class AggregateConfig:
    n_frames: int = 5            # expected burst length (1 s at 5 fps); informational
    min_valid: int = 3           # fewer usable frames -> retry_capture
    pass_ratio: float = 0.8      # share of usable frames that must vote "none" (4 of 5)
    fail_votes: int = 2          # >= this many "glasses" frames -> remove_glasses
    min_det_score: float = 0.6   # SCRFD score below this -> frame rejected (low_det)
    min_blur: float = 30.0       # Laplacian variance on the 160px crop; tune from logs
                                 # (sharp web photos: 300-1000; motion-blurred crops that
                                 #  flipped glasses->none in a smoke test: 20-25)
    min_eye_dist: float = 24.0   # px between the eyes in the submitted frame (too_far)
    # What a valid frame votes on. "eyewear" = P(eyeglasses)+P(sunglasses) — sunglasses stop
    # the process too (verification gate). "eyeglasses" = legacy P(eyeglasses) with the
    # calibrated t_low/t_high band from threshold.json.
    vote_on: str = "eyewear"
    eyewear_t_low: float = 0.2   # eyewear below this -> "none" vote
    eyewear_t_high: float = 0.5  # eyewear at/above this -> "glasses" vote; between -> unsure

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
            setattr(self, k, type(cur)(v))

    def to_dict(self) -> dict:
        return asdict(self)


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
