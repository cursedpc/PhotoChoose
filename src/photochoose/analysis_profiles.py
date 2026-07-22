from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

ANALYSIS_VERSION = "profiles-v2-eye-visibility"


@dataclass(frozen=True)
class AnalysisProfile:
    """Named, explainable thresholds used by automatic analysis."""

    key: str
    label: str
    description: str
    severe_focus_threshold: float
    focus_review_threshold: float
    dark_brightness_threshold: float
    bright_brightness_threshold: float
    clipped_exposure_threshold: float
    face_focus_threshold: float
    face_confidence_threshold: float
    eye_open_score_threshold: float
    object_focus_threshold: float
    relative_reject_score_gap: float


SAFE_PROFILE = AnalysisProfile(
    key="safe",
    label="Safe",
    description="Conservative separation; uncertain photographs stay in Review.",
    severe_focus_threshold=8.0,
    focus_review_threshold=30.0,
    dark_brightness_threshold=0.08,
    bright_brightness_threshold=0.92,
    clipped_exposure_threshold=0.75,
    face_focus_threshold=20.0,
    face_confidence_threshold=0.0,
    eye_open_score_threshold=0.0,
    object_focus_threshold=20.0,
    relative_reject_score_gap=25.0,
)

BALANCED_PROFILE = AnalysisProfile(
    key="balanced",
    label="Balanced",
    description="Flags moderately weak focus, exposure, face, and eye evidence.",
    severe_focus_threshold=12.0,
    focus_review_threshold=42.0,
    dark_brightness_threshold=0.10,
    bright_brightness_threshold=0.90,
    clipped_exposure_threshold=0.65,
    face_focus_threshold=30.0,
    face_confidence_threshold=0.68,
    eye_open_score_threshold=0.42,
    object_focus_threshold=30.0,
    relative_reject_score_gap=20.0,
)

STRICT_PROFILE = AnalysisProfile(
    key="strict",
    label="Strict",
    description="Reviews borderline frames and separates weak burst alternatives sooner.",
    severe_focus_threshold=18.0,
    focus_review_threshold=60.0,
    dark_brightness_threshold=0.14,
    bright_brightness_threshold=0.86,
    clipped_exposure_threshold=0.50,
    face_focus_threshold=45.0,
    face_confidence_threshold=0.78,
    eye_open_score_threshold=0.56,
    object_focus_threshold=45.0,
    relative_reject_score_gap=14.0,
)

ANALYSIS_PROFILES: Tuple[AnalysisProfile, ...] = (
    SAFE_PROFILE,
    BALANCED_PROFILE,
    STRICT_PROFILE,
)
_PROFILE_BY_KEY: Dict[str, AnalysisProfile] = {
    profile.key: profile for profile in ANALYSIS_PROFILES
}


def get_analysis_profile(profile: Union[str, AnalysisProfile]) -> AnalysisProfile:
    if isinstance(profile, AnalysisProfile):
        return profile
    try:
        return _PROFILE_BY_KEY[profile]
    except KeyError as exc:
        raise ValueError("Unknown analysis profile: {0}".format(profile)) from exc
