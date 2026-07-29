from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from photochoose.analysis_profiles import (
    ANALYSIS_VERSION,
    SAFE_PROFILE,
    AnalysisProfile,
    get_analysis_profile,
)
from photochoose.database import PhotoDatabase
from photochoose.image_loader import ImageDecodeError, decode_analysis_bgr
from photochoose.models import Decision, FaceEvidence, ObjectEvidence, Photo, PhotoView
from photochoose.object_models import GeneralSubjectAnalyzer
from photochoose.vision_models import ModelSubjectAnalyzer

MAX_ANALYSIS_DIMENSION = 1600
MAX_FACE_DETECTION_DIMENSION = 900
NEAR_DUPLICATE_DISTANCE = 8
BURST_SIMILARITY_DISTANCE = 20
BURST_GAP_SECONDS = 2.0
GROUP_COMPARISON_WINDOW = 12
MIN_BEST_HASH_DISTANCE = NEAR_DUPLICATE_DISTANCE + 1


@dataclass(frozen=True)
class AnalysisResult:
    decision: Decision
    focus_score: Optional[float]
    brightness: Optional[float]
    clipped_dark: Optional[float]
    clipped_bright: Optional[float]
    content_hash: Optional[str]
    perceptual_hash: Optional[str]
    face_count: int
    eye_count: int
    subject_focus_score: Optional[float]
    reasons: Tuple[str, ...]
    closed_eye_count: int = 0
    uncertain_eye_count: int = 0
    obscured_eye_count: int = 0
    subject_model: str = "none"
    face_confidence: Optional[float] = None
    eye_open_score: Optional[float] = None
    eye_focus_score: Optional[float] = None
    important_face_count: int = 0
    important_open_eye_count: int = 0
    important_closed_eye_count: int = 0
    important_uncertain_eye_count: int = 0
    important_obscured_eye_count: int = 0
    subject_regions: Tuple[FaceEvidence, ...] = ()
    object_count: int = 0
    object_model: str = "none"
    primary_subject_focus_score: Optional[float] = None
    subject_objects: Tuple[ObjectEvidence, ...] = ()
    duplicate_of_id: Optional[int] = None
    status: str = "complete"


class BaselineAnalyzer:
    """CPU analysis for technical failures, faces, similarity, and group ranking."""

    def __init__(
        self,
        database: PhotoDatabase,
        profile: Union[AnalysisProfile, str] = SAFE_PROFILE,
    ) -> None:
        self.database = database
        self.profile = get_analysis_profile(profile)
        self.subject_analyzer = ModelSubjectAnalyzer()
        self.object_analyzer = GeneralSubjectAnalyzer()
        cascade_root = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            str(cascade_root / "haarcascade_eye_tree_eyeglasses.xml")
        )

    def analyze_project(
        self,
        project_id: int,
        progress: Optional[Callable[[int, int], None]] = None,
        pending_only: bool = False,
    ) -> Dict[str, int]:
        photos = sorted(
            self.database.list_photos(project_id, PhotoView.ALL),
            key=lambda photo: (
                len(Path(photo.relative_path).parts),
                len(photo.relative_path),
                photo.relative_path.casefold(),
            ),
        )
        stored = {
            photo.id: self._stored_result(photo)
            for photo in photos
            if pending_only and photo.analysis_status != "pending"
        }
        hashes: Dict[str, Tuple[Photo, AnalysisResult]] = {
            result.content_hash: (photo, result)
            for photo in photos
            if (result := stored.get(photo.id)) is not None and result.content_hash
        }
        analyzed: List[Tuple[Photo, AnalysisResult]] = []
        pending_count = len(photos) - len(stored)
        completed = 0

        if progress is not None:
            progress(0, pending_count)
        for photo in photos:
            result = stored.get(photo.id)
            if result is not None:
                analyzed.append((photo, result))
                continue
            result = self.analyze_photo(photo, hashes)
            self.database.apply_analysis(
                photo.id,
                result.decision,
                result.focus_score,
                result.brightness,
                result.clipped_dark,
                result.clipped_bright,
                result.content_hash,
                result.reasons,
                result.duplicate_of_id,
                result.status,
                self.profile.key,
                ANALYSIS_VERSION,
            )
            self.database.apply_subject_analysis(
                photo.id,
                result.perceptual_hash,
                result.face_count,
                result.eye_count,
                result.closed_eye_count,
                result.uncertain_eye_count,
                result.subject_focus_score,
                result.subject_model,
                result.face_confidence,
                result.eye_open_score,
                result.eye_focus_score,
                result.subject_regions,
            )
            self.database.apply_object_analysis(
                photo.id,
                result.object_count,
                result.object_model,
                result.primary_subject_focus_score,
                result.subject_objects,
            )
            if result.content_hash and result.content_hash not in hashes:
                hashes[result.content_hash] = (photo, result)
            analyzed.append((photo, result))
            completed += 1
            if progress is not None:
                progress(completed, pending_count)

        self._group_and_select_best(project_id, analyzed)
        return self.database.counts(project_id)

    @staticmethod
    def _stored_result(photo: Photo) -> AnalysisResult:
        important_eyes = [
            eye
            for region in photo.subject_regions
            if region.important
            for eye in region.eyes
        ]
        return AnalysisResult(
            decision=photo.decision,
            focus_score=photo.focus_score,
            brightness=photo.brightness,
            clipped_dark=photo.clipped_dark,
            clipped_bright=photo.clipped_bright,
            content_hash=photo.content_hash,
            perceptual_hash=photo.perceptual_hash,
            face_count=photo.face_count,
            eye_count=photo.eye_count,
            subject_focus_score=photo.subject_focus_score,
            reasons=photo.auto_reasons,
            closed_eye_count=photo.closed_eye_count,
            uncertain_eye_count=photo.uncertain_eye_count,
            obscured_eye_count=photo.obscured_eye_count,
            subject_model=photo.subject_model,
            face_confidence=photo.face_confidence,
            eye_open_score=photo.eye_open_score,
            eye_focus_score=photo.eye_focus_score,
            important_face_count=photo.important_face_count,
            important_open_eye_count=sum(eye.state == "open" for eye in important_eyes),
            important_closed_eye_count=sum(
                eye.state == "closed" for eye in important_eyes
            ),
            important_uncertain_eye_count=sum(
                eye.state == "uncertain" for eye in important_eyes
            ),
            important_obscured_eye_count=sum(
                eye.state == "obscured" for eye in important_eyes
            ),
            subject_regions=photo.subject_regions,
            object_count=photo.object_count,
            object_model=photo.object_model,
            primary_subject_focus_score=photo.primary_subject_focus_score,
            subject_objects=photo.subject_objects,
            duplicate_of_id=photo.duplicate_of_id,
            status=photo.analysis_status,
        )

    def analyze_photo(
        self,
        photo: Photo,
        known_hashes: Dict[str, Tuple[Photo, AnalysisResult]],
    ) -> AnalysisResult:
        analysis_path = Path(photo.display_path)
        if photo.raw_count and (
            photo.preview_absolute_path is None or not analysis_path.is_file()
        ):
            return AnalysisResult(
                decision=Decision.REVIEW,
                focus_score=None,
                brightness=None,
                clipped_dark=None,
                clipped_bright=None,
                content_hash=None,
                perceptual_hash=None,
                face_count=0,
                eye_count=0,
                subject_focus_score=None,
                reasons=(
                    "RAW file indexed safely, but no usable local preview was found",
                ),
                status="preview-unavailable",
            )
        try:
            encoded = analysis_path.read_bytes()
        except OSError as exc:
            return self._error_result("Unreadable file: {0}".format(exc))

        content_hash = hashlib.sha256(encoded).hexdigest()
        duplicate = known_hashes.get(content_hash)
        if duplicate is not None:
            original, original_result = duplicate
            return AnalysisResult(
                decision=Decision.REJECT,
                focus_score=original_result.focus_score,
                brightness=original_result.brightness,
                clipped_dark=original_result.clipped_dark,
                clipped_bright=original_result.clipped_bright,
                content_hash=content_hash,
                perceptual_hash=original_result.perceptual_hash,
                face_count=original_result.face_count,
                eye_count=original_result.eye_count,
                subject_focus_score=original_result.subject_focus_score,
                reasons=("Exact duplicate of {0}".format(original.filename),),
                closed_eye_count=original_result.closed_eye_count,
                uncertain_eye_count=original_result.uncertain_eye_count,
                obscured_eye_count=original_result.obscured_eye_count,
                subject_model=original_result.subject_model,
                face_confidence=original_result.face_confidence,
                eye_open_score=original_result.eye_open_score,
                eye_focus_score=original_result.eye_focus_score,
                important_face_count=original_result.important_face_count,
                important_open_eye_count=original_result.important_open_eye_count,
                important_closed_eye_count=original_result.important_closed_eye_count,
                important_uncertain_eye_count=(
                    original_result.important_uncertain_eye_count
                ),
                important_obscured_eye_count=(
                    original_result.important_obscured_eye_count
                ),
                subject_regions=original_result.subject_regions,
                object_count=original_result.object_count,
                object_model=original_result.object_model,
                primary_subject_focus_score=(
                    original_result.primary_subject_focus_score
                ),
                subject_objects=original_result.subject_objects,
                duplicate_of_id=original.id,
            )

        try:
            image = decode_analysis_bgr(analysis_path, encoded)
        except ImageDecodeError:
            return self._error_result("Image data cannot be decoded", content_hash=content_hash)

        image = self._downscale(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean() / 255.0)
        clipped_dark = float(np.count_nonzero(gray <= 5) / gray.size)
        clipped_bright = float(np.count_nonzero(gray >= 250) / gray.size)
        perceptual_hash = self._perceptual_hash(gray)
        subject_result = self.subject_analyzer.analyze(image)
        if subject_result is None:
            face_count, eye_count, subject_focus = self._analyze_faces(gray)
            closed_eye_count = 0
            uncertain_eye_count = max(0, face_count * 2 - eye_count)
            obscured_eye_count = 0
            subject_model = "haar-fallback"
            face_confidence = None
            eye_open_score = None
            eye_focus_score = None
            important_face_count = face_count
            important_open_eye_count = eye_count
            important_closed_eye_count = closed_eye_count
            important_uncertain_eye_count = uncertain_eye_count
            important_obscured_eye_count = 0
            subject_regions = ()
        else:
            face_count = subject_result.face_count
            eye_count = subject_result.open_eye_count
            closed_eye_count = subject_result.closed_eye_count
            uncertain_eye_count = subject_result.uncertain_eye_count
            obscured_eye_count = subject_result.obscured_eye_count
            subject_focus = subject_result.subject_focus_score
            subject_model = subject_result.model_name
            face_confidence = subject_result.face_confidence
            eye_open_score = subject_result.eye_open_score
            eye_focus_score = subject_result.eye_focus_score
            important_face_count = subject_result.important_face_count
            important_open_eye_count = subject_result.important_open_eye_count
            important_closed_eye_count = subject_result.important_closed_eye_count
            important_uncertain_eye_count = subject_result.important_uncertain_eye_count
            important_obscured_eye_count = subject_result.important_obscured_eye_count
            subject_regions = subject_result.regions

        object_result = self.object_analyzer.analyze(image)

        decision, reasons = self._classify(
            focus_score,
            brightness,
            clipped_dark,
            clipped_bright,
            face_count,
            eye_count,
            subject_focus,
            closed_eye_count,
            uncertain_eye_count,
            important_face_count,
            important_open_eye_count,
            important_closed_eye_count,
            important_uncertain_eye_count,
            object_result.primary_focus_score,
            object_result.model_name,
            face_confidence,
            eye_open_score,
            self.profile,
            obscured_eye_count=obscured_eye_count,
            important_obscured_eye_count=important_obscured_eye_count,
        )
        return AnalysisResult(
            decision=decision,
            focus_score=round(focus_score, 2),
            brightness=round(brightness, 4),
            clipped_dark=round(clipped_dark, 4),
            clipped_bright=round(clipped_bright, 4),
            content_hash=content_hash,
            perceptual_hash=perceptual_hash,
            face_count=face_count,
            eye_count=eye_count,
            subject_focus_score=(round(subject_focus, 2) if subject_focus is not None else None),
            reasons=reasons,
            closed_eye_count=closed_eye_count,
            uncertain_eye_count=uncertain_eye_count,
            obscured_eye_count=obscured_eye_count,
            subject_model=subject_model,
            face_confidence=(
                round(face_confidence, 4) if face_confidence is not None else None
            ),
            eye_open_score=(
                round(eye_open_score, 4) if eye_open_score is not None else None
            ),
            eye_focus_score=(
                round(eye_focus_score, 2) if eye_focus_score is not None else None
            ),
            important_face_count=important_face_count,
            important_open_eye_count=important_open_eye_count,
            important_closed_eye_count=important_closed_eye_count,
            important_uncertain_eye_count=important_uncertain_eye_count,
            important_obscured_eye_count=important_obscured_eye_count,
            subject_regions=subject_regions,
            object_count=object_result.object_count,
            object_model=object_result.model_name,
            primary_subject_focus_score=(
                round(object_result.primary_focus_score, 2)
                if object_result.primary_focus_score is not None
                else None
            ),
            subject_objects=object_result.objects,
        )

    @staticmethod
    def _error_result(message: str, content_hash: Optional[str] = None) -> AnalysisResult:
        return AnalysisResult(
            decision=Decision.REJECT,
            focus_score=None,
            brightness=None,
            clipped_dark=None,
            clipped_bright=None,
            content_hash=content_hash,
            perceptual_hash=None,
            face_count=0,
            eye_count=0,
            subject_focus_score=None,
            reasons=(message,),
            status="error",
        )

    def _analyze_faces(self, gray: np.ndarray) -> Tuple[int, int, Optional[float]]:
        if self.face_cascade.empty() or self.eye_cascade.empty():
            return 0, 0, None

        detection = gray
        scale = 1.0
        longest = max(gray.shape[:2])
        if longest > MAX_FACE_DETECTION_DIMENSION:
            scale = MAX_FACE_DETECTION_DIMENSION / float(longest)
            detection = cv2.resize(
                gray,
                (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )

        faces = self.face_cascade.detectMultiScale(
            detection,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(45, 45),
        )
        eye_count = 0
        focus_scores = []
        for detected_x, detected_y, detected_w, detected_h in faces:
            x = int(detected_x / scale)
            y = int(detected_y / scale)
            width = max(1, int(detected_w / scale))
            height = max(1, int(detected_h / scale))
            face = gray[y : y + height, x : x + width]
            if face.size == 0:
                continue
            focus_scores.append(float(cv2.Laplacian(face, cv2.CV_64F).var()))
            upper_face = face[: max(1, int(face.shape[0] * 0.68)), :]
            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(max(10, width // 8), max(8, height // 10)),
            )
            eye_count += min(2, len(eyes))

        return len(faces), eye_count, min(focus_scores) if focus_scores else None

    def _group_and_select_best(
        self,
        project_id: int,
        analyzed: List[Tuple[Photo, AnalysisResult]],
    ) -> None:
        self.database.reset_automatic_groups(project_id)
        usable = [item for item in analyzed if item[1].perceptual_hash is not None]
        usable.sort(
            key=lambda item: (
                item[0].captured_at is None,
                item[0].captured_at or "",
                item[0].relative_path.casefold(),
            )
        )
        parents = list(range(len(usable)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        first_content_index = {}
        for index, (_photo, result) in enumerate(usable):
            if result.content_hash in first_content_index:
                union(first_content_index[result.content_hash], index)
            elif result.content_hash:
                first_content_index[result.content_hash] = index

        first_burst_index = {}
        for index, (photo, _result) in enumerate(usable):
            if not photo.burst_id:
                continue
            burst_key = (
                Path(photo.relative_path).parent.as_posix().casefold(),
                photo.burst_id.casefold(),
            )
            if burst_key in first_burst_index:
                union(first_burst_index[burst_key], index)
            else:
                first_burst_index[burst_key] = index

        for left_index, (left_photo, left_result) in enumerate(usable):
            end = min(len(usable), left_index + GROUP_COMPARISON_WINDOW + 1)
            for right_index in range(left_index + 1, end):
                right_photo, right_result = usable[right_index]
                same_folder = (
                    Path(left_photo.relative_path).parent == Path(right_photo.relative_path).parent
                )
                if not same_folder:
                    continue
                distance = self._hash_distance(
                    left_result.perceptual_hash, right_result.perceptual_hash
                )
                capture_gap = self._capture_gap_seconds(left_photo, right_photo)
                same_camera = self._same_camera(left_photo, right_photo)
                timed_burst = (
                    capture_gap is not None
                    and capture_gap <= BURST_GAP_SECONDS
                    and same_camera
                    and distance <= BURST_SIMILARITY_DISTANCE
                )
                visual_fallback = capture_gap is None and distance <= NEAR_DUPLICATE_DISTANCE
                if timed_burst or visual_fallback:
                    union(left_index, right_index)

        groups: Dict[int, List[Tuple[Photo, AnalysisResult]]] = {}
        for index, item in enumerate(usable):
            groups.setdefault(find(index), []).append(item)

        current_photos = {
            photo.id: photo for photo in self.database.list_photos(project_id, PhotoView.ALL)
        }
        group_number = 0
        for items in groups.values():
            if len(items) < 2:
                continue
            group_number += 1
            group_prefix, group_reason = self._group_description(items)
            group_id = "{0}{1:04d}".format(group_prefix, group_number)
            scored = []
            for original_photo, result in items:
                current = current_photos[original_photo.id]
                score = self._best_score(current.decision, result)
                self.database.set_similarity_group(
                    current.id, group_id, score, group_reason
                )
                scored.append((current, score))

            if any(photo.best_source == "manual" and photo.is_best for photo, _ in scored):
                continue
            candidates = [
                (photo, score)
                for photo, score in scored
                if photo.decision != Decision.REJECT and photo.best_source != "manual"
            ]
            if candidates:
                winner, winner_score = max(candidates, key=lambda item: item[1])
                for photo, score in scored:
                    score_gap = winner_score - score
                    if (
                        photo.id != winner.id
                        and score_gap >= self.profile.relative_reject_score_gap
                    ):
                        self.database.reject_weaker_group_photo(
                            photo.id,
                            group_id,
                            winner.filename,
                            score_gap,
                        )

        project = self.database.get_project(project_id)
        if project is not None:
            self.database.select_automatic_best(
                project_id,
                project.best_count_per_group,
                MIN_BEST_HASH_DISTANCE,
            )

    @staticmethod
    def _capture_gap_seconds(left: Photo, right: Photo) -> Optional[float]:
        if not left.captured_at or not right.captured_at:
            return None
        try:
            left_time = datetime.fromisoformat(left.captured_at)
            right_time = datetime.fromisoformat(right.captured_at)
        except ValueError:
            return None
        return abs((right_time - left_time).total_seconds())

    @staticmethod
    def _same_camera(left: Photo, right: Photo) -> bool:
        if left.camera_model and right.camera_model:
            return left.camera_model.casefold() == right.camera_model.casefold()
        return True

    @classmethod
    def _group_description(
        cls, items: List[Tuple[Photo, AnalysisResult]]
    ) -> Tuple[str, str]:
        burst_ids = {photo.burst_id.casefold() for photo, _result in items if photo.burst_id}
        if len(burst_ids) == 1:
            return "B", "Burst metadata · {0} frames".format(len(items))
        timestamps = []
        for photo, _result in items:
            if not photo.captured_at:
                continue
            try:
                timestamps.append(datetime.fromisoformat(photo.captured_at))
            except ValueError:
                continue
        if len(timestamps) >= 2:
            span = (max(timestamps) - min(timestamps)).total_seconds()
            return "B", "Burst · {0} frames over {1:.2f}s".format(len(items), span)
        return "G", "Visually similar frames · {0} photographs".format(len(items))

    @staticmethod
    def _perceptual_hash(gray: np.ndarray) -> str:
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        frequency = cv2.dct(np.float32(resized))[:8, :8]
        median = float(np.median(frequency[1:, :]))
        bits = frequency > median
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return "{0:016x}".format(value)

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        return bin(int(left, 16) ^ int(right, 16)).count("1")

    @staticmethod
    def _best_score(decision: Decision, result: AnalysisResult) -> float:
        if decision == Decision.REJECT:
            return -100.0
        focus = min(1.0, math.log1p(result.focus_score or 0.0) / math.log(501.0))
        exposure = max(0.0, 1.0 - abs((result.brightness or 0.5) - 0.5) / 0.5)
        important_face_count = result.important_face_count or result.face_count
        important_open_eye_count = (
            result.important_open_eye_count
            if result.important_face_count
            else result.eye_count
        )
        important_closed_eye_count = (
            result.important_closed_eye_count
            if result.important_face_count
            else result.closed_eye_count
        )
        important_obscured_eye_count = (
            result.important_obscured_eye_count
            if result.important_face_count
            else result.obscured_eye_count
        )
        if important_face_count:
            subject = min(
                1.0,
                math.log1p(result.subject_focus_score or 0.0) / math.log(501.0),
            )
            expected_eyes = float(important_face_count * 2)
            eyes = min(1.0, important_open_eye_count / expected_eyes)
            eye_focus = min(
                1.0,
                math.log1p(result.eye_focus_score or 0.0) / math.log(1001.0),
            )
            closed_eye_penalty = (
                min(1.0, important_closed_eye_count / expected_eyes) * 20.0
            )
            obscured_eye_penalty = (
                min(1.0, important_obscured_eye_count / expected_eyes) * 5.0
            )
        else:
            subject = focus
            eyes = 0.5
            eye_focus = focus
            closed_eye_penalty = 0.0
            obscured_eye_penalty = 0.0
        review_penalty = 10.0 if decision == Decision.REVIEW else 0.0
        object_adjustment = 0.0
        if result.primary_subject_focus_score is not None:
            object_focus = min(
                1.0,
                math.log1p(result.primary_subject_focus_score) / math.log(501.0),
            )
            object_adjustment = (object_focus - 0.5) * 12.0
        return round(
            focus * 30.0
            + exposure * 15.0
            + subject * 25.0
            + eyes * 15.0
            + eye_focus * 15.0
            - review_penalty
            - closed_eye_penalty
            - obscured_eye_penalty
            + object_adjustment,
            2,
        )

    @staticmethod
    def _downscale(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= MAX_ANALYSIS_DIMENSION:
            return image
        scale = MAX_ANALYSIS_DIMENSION / float(longest)
        return cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _classify(
        focus_score: float,
        brightness: float,
        clipped_dark: float,
        clipped_bright: float,
        face_count: int,
        eye_count: int,
        subject_focus: Optional[float],
        closed_eye_count: int = 0,
        uncertain_eye_count: int = 0,
        important_face_count: Optional[int] = None,
        important_eye_count: Optional[int] = None,
        important_closed_eye_count: Optional[int] = None,
        important_uncertain_eye_count: Optional[int] = None,
        primary_subject_focus: Optional[float] = None,
        object_model: str = "none",
        face_confidence: Optional[float] = None,
        eye_open_score: Optional[float] = None,
        profile: AnalysisProfile = SAFE_PROFILE,
        obscured_eye_count: int = 0,
        important_obscured_eye_count: Optional[int] = None,
    ) -> Tuple[Decision, Tuple[str, ...]]:
        if clipped_dark >= 0.98:
            return Decision.REVIEW, ("Almost entirely black",)
        if clipped_bright >= 0.98:
            return Decision.REVIEW, ("Almost entirely white or clipped",)

        reasons = []
        if focus_score < profile.severe_focus_threshold:
            reasons.append("Very low detail; possible severe blur")
        elif focus_score < profile.focus_review_threshold:
            reasons.append("Possibly soft or out of focus")

        if (
            brightness < profile.dark_brightness_threshold
            or clipped_dark >= profile.clipped_exposure_threshold
        ):
            reasons.append("Very dark exposure")
        elif (
            brightness > profile.bright_brightness_threshold
            or clipped_bright >= profile.clipped_exposure_threshold
        ):
            reasons.append("Very bright exposure")

        considered_faces = (
            face_count if important_face_count is None else important_face_count
        )
        considered_open_eyes = (
            eye_count if important_eye_count is None else important_eye_count
        )
        considered_closed_eyes = (
            closed_eye_count
            if important_closed_eye_count is None
            else important_closed_eye_count
        )
        considered_uncertain_eyes = (
            uncertain_eye_count
            if important_uncertain_eye_count is None
            else important_uncertain_eye_count
        )
        considered_obscured_eyes = (
            obscured_eye_count
            if important_obscured_eye_count is None
            else important_obscured_eye_count
        )
        if considered_faces:
            if (
                subject_focus is not None
                and subject_focus < profile.face_focus_threshold
            ):
                reasons.append("At least one important face appears soft")
            if (
                face_confidence is not None
                and face_confidence < profile.face_confidence_threshold
            ):
                reasons.append("Important face detection is not fully confident")
            if considered_closed_eyes:
                reasons.append(
                    "{0} detected eye{1} appear{2} closed".format(
                        considered_closed_eyes,
                        "" if considered_closed_eyes == 1 else "s",
                        "s" if considered_closed_eyes == 1 else "",
                    )
                )
            if considered_obscured_eyes:
                reasons.append(
                    "{0} important eye{1} appear{2} obscured or unresolved".format(
                        considered_obscured_eyes,
                        "" if considered_obscured_eyes == 1 else "s",
                        "s" if considered_obscured_eyes == 1 else "",
                    )
                )
            if (
                considered_uncertain_eyes
                or considered_open_eyes
                + considered_closed_eyes
                + considered_obscured_eyes
                < considered_faces * 2
            ):
                reasons.append("Eyes are not confidently visible on every important face")
            elif (
                eye_open_score is not None
                and eye_open_score < profile.eye_open_score_threshold
            ):
                reasons.append("Eye openness is borderline on an important face")

        if (
            object_model.startswith("nanodet")
            and primary_subject_focus is not None
            and primary_subject_focus < profile.object_focus_threshold
        ):
            reasons.append("At least one primary detected subject appears soft")

        if reasons:
            return Decision.REVIEW, tuple(reasons)
        return Decision.KEEP, ("Passed technical and subject checks",)
