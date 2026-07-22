from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Decision(str, Enum):
    UNREVIEWED = "unreviewed"
    KEEP = "keep"
    REVIEW = "review"
    REJECT = "reject"


class PhotoView(str, Enum):
    ALL = "all"
    BEST = "best"
    KEEP = "keep"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class Project:
    id: int
    name: str
    root_path: str
    analysis_profile: str = "safe"
    best_count_per_group: int = 1


@dataclass(frozen=True)
class EyeEvidence:
    """One eye measurement in normalized image coordinates."""

    state: str
    aperture: Optional[float]
    openness_score: Optional[float]
    focus_score: Optional[float]
    landmarks: Tuple[Tuple[float, float], ...] = ()
    visibility_reason: Optional[str] = None


@dataclass(frozen=True)
class FaceEvidence:
    """Visible evidence for one detected face."""

    bounds: Tuple[float, float, float, float]
    confidence: float
    focus_score: Optional[float]
    important: bool
    eyes: Tuple[EyeEvidence, ...] = ()


@dataclass(frozen=True)
class ObjectEvidence:
    """One general-subject region in normalized image coordinates."""

    label: str
    bounds: Tuple[float, float, float, float]
    confidence: float
    focus_score: Optional[float]
    importance: str


@dataclass(frozen=True)
class Photo:
    id: int
    project_id: int
    relative_path: str
    absolute_path: str
    decision: Decision
    is_best: bool
    decision_source: str = "none"
    analysis_status: str = "pending"
    analysis_profile: Optional[str] = None
    analysis_version: Optional[str] = None
    focus_score: Optional[float] = None
    brightness: Optional[float] = None
    clipped_dark: Optional[float] = None
    clipped_bright: Optional[float] = None
    auto_reasons: Tuple[str, ...] = ()
    duplicate_of_id: Optional[int] = None
    best_source: str = "none"
    perceptual_hash: Optional[str] = None
    face_count: int = 0
    eye_count: int = 0
    subject_focus_score: Optional[float] = None
    similarity_group: Optional[str] = None
    best_score: Optional[float] = None
    group_reason: Optional[str] = None
    captured_at: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    iso: Optional[int] = None
    exposure_seconds: Optional[float] = None
    aperture: Optional[float] = None
    focal_length: Optional[float] = None
    preview_absolute_path: Optional[str] = None
    asset_paths: Tuple[str, ...] = ()
    raw_count: int = 0
    sidecar_count: int = 0
    closed_eye_count: int = 0
    uncertain_eye_count: int = 0
    subject_model: str = "none"
    face_confidence: Optional[float] = None
    eye_open_score: Optional[float] = None
    eye_focus_score: Optional[float] = None
    subject_regions: Tuple[FaceEvidence, ...] = ()
    object_count: int = 0
    object_model: str = "none"
    primary_subject_focus_score: Optional[float] = None
    subject_objects: Tuple[ObjectEvidence, ...] = ()

    @property
    def filename(self) -> str:
        return self.relative_path.rsplit("/", 1)[-1]

    @property
    def important_face_count(self) -> int:
        return sum(region.important for region in self.subject_regions)

    @property
    def primary_object_count(self) -> int:
        return sum(region.importance == "primary" for region in self.subject_objects)

    @property
    def obscured_eye_count(self) -> int:
        return sum(
            eye.state == "obscured"
            for region in self.subject_regions
            for eye in region.eyes
        )

    @property
    def important_obscured_eye_count(self) -> int:
        return sum(
            eye.state == "obscured"
            for region in self.subject_regions
            if region.important
            for eye in region.eyes
        )

    @property
    def eye_visibility_reasons(self) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                eye.visibility_reason
                for region in self.subject_regions
                for eye in region.eyes
                if eye.visibility_reason
            )
        )

    @property
    def display_path(self) -> str:
        return self.preview_absolute_path or self.absolute_path

    @property
    def has_displayable_preview(self) -> bool:
        return self.preview_absolute_path is not None or self.raw_count == 0

    @property
    def file_summary(self) -> str:
        parts = []
        if self.raw_count:
            parts.append("{0} RAW".format(self.raw_count))
        preview_count = max(0, len(self.asset_paths) - self.raw_count - self.sidecar_count)
        if preview_count:
            parts.append("{0} preview".format(preview_count))
        if self.sidecar_count:
            parts.append("{0} sidecar".format(self.sidecar_count))
        return " + ".join(parts) or "1 image"

    @property
    def asset_badge(self) -> Optional[str]:
        if not self.raw_count:
            return "XMP" if self.sidecar_count else None
        preview_count = max(0, len(self.asset_paths) - self.raw_count - self.sidecar_count)
        label = "RAW+JPG" if preview_count else "RAW"
        return label + ("+XMP" if self.sidecar_count else "")
