from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from photochoose.models import EyeEvidence, FaceEvidence

MODEL_DIRECTORY = Path(__file__).with_name("model_assets")
FACE_MODEL_PATH = MODEL_DIRECTORY / "face_detection_yunet_2023mar.onnx"
FACE_LANDMARK_MODEL_PATH = MODEL_DIRECTORY / "face_landmarks_detector.tflite"
MAX_DETECTION_DIMENSION = 1280
CLOSED_EYE_APERTURE = 0.14
OPEN_EYE_APERTURE = 0.19
MIN_VISIBLE_EYE_WIDTH = 6.0
MIN_VISIBLE_EYE_CONTRAST = 6.0
MIN_VISIBLE_EYE_FOCUS = 8.0
MAX_DARK_EYE_FRACTION = 0.78
MAX_BRIGHT_EYE_FRACTION = 0.85
RIGHT_EYE_LANDMARKS = (33, 160, 158, 133, 153, 144)
LEFT_EYE_LANDMARKS = (362, 385, 387, 263, 373, 380)


@dataclass(frozen=True)
class SubjectModelResult:
    face_count: int
    open_eye_count: int
    closed_eye_count: int
    uncertain_eye_count: int
    obscured_eye_count: int
    subject_focus_score: Optional[float]
    face_confidence: Optional[float]
    eye_open_score: Optional[float]
    model_name: str
    eye_focus_score: Optional[float] = None
    important_face_count: int = 0
    important_open_eye_count: int = 0
    important_closed_eye_count: int = 0
    important_uncertain_eye_count: int = 0
    important_obscured_eye_count: int = 0
    regions: Tuple[FaceEvidence, ...] = ()


@dataclass(frozen=True)
class EyeLandmarkResult:
    state: str
    aperture: Optional[float]
    openness_score: Optional[float]
    landmarks: Tuple[Tuple[float, float], ...] = ()


class FaceLandmarkEyeAnalyzer:
    """Measure eyelid geometry using the local MediaPipe face-mesh model."""

    def __init__(
        self,
        model_path: Path = FACE_LANDMARK_MODEL_PATH,
        inference: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._inference = inference
        self.error: Optional[str] = None
        if self._inference is None:
            try:
                from ai_edge_litert.interpreter import Interpreter

                interpreter = Interpreter(model_path=str(model_path))
                interpreter.allocate_tensors()
                input_index = interpreter.get_input_details()[0]["index"]
                landmark_output_index = next(
                    output["index"]
                    for output in interpreter.get_output_details()
                    if output["shape"][-1] == 1434
                )

                def infer(tensor: np.ndarray) -> np.ndarray:
                    interpreter.set_tensor(input_index, tensor)
                    interpreter.invoke()
                    return interpreter.get_tensor(landmark_output_index)

                self._inference = infer
            except (ImportError, RuntimeError, StopIteration, OSError) as exc:
                self.error = str(exc)

    @property
    def available(self) -> bool:
        return self._inference is not None

    def analyze(self, face_image: np.ndarray) -> Tuple[EyeLandmarkResult, ...]:
        if self._inference is None or face_image.size == 0:
            return self._uncertain_pair()
        rgb = cv2.cvtColor(
            cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        tensor = (rgb.astype(np.float32) / 255.0)[np.newaxis, ...]
        try:
            landmarks = np.asarray(self._inference(tensor), dtype=np.float32).reshape(-1, 3)
        except (RuntimeError, ValueError, cv2.error) as exc:
            self.error = str(exc)
            self._inference = None
            return self._uncertain_pair()
        if len(landmarks) < 478:
            return self._uncertain_pair()
        return tuple(
            self._classify_eye(landmarks[np.asarray(indices), :2])
            for indices in (RIGHT_EYE_LANDMARKS, LEFT_EYE_LANDMARKS)
        )

    @classmethod
    def _classify_eye(cls, eye: np.ndarray) -> EyeLandmarkResult:
        aperture = cls._eye_aperture(eye)
        if aperture is None:
            return EyeLandmarkResult("uncertain", None, None)
        openness = min(
            1.0,
            max(
                0.0,
                (aperture - CLOSED_EYE_APERTURE)
                / (OPEN_EYE_APERTURE - CLOSED_EYE_APERTURE),
            ),
        )
        if aperture <= CLOSED_EYE_APERTURE:
            state = "closed"
        elif aperture >= OPEN_EYE_APERTURE:
            state = "open"
        else:
            state = "uncertain"
        return EyeLandmarkResult(
            state,
            aperture,
            openness,
            tuple((float(point[0]), float(point[1])) for point in eye),
        )

    @staticmethod
    def _eye_aperture(eye: np.ndarray) -> Optional[float]:
        if eye.shape != (6, 2) or not np.isfinite(eye).all():
            return None
        width = float(np.linalg.norm(eye[0] - eye[3]))
        if width < 8.0:
            return None
        height = float(np.linalg.norm(eye[1] - eye[5])) + float(
            np.linalg.norm(eye[2] - eye[4])
        )
        return height / (2.0 * width)

    @staticmethod
    def _uncertain_pair() -> Tuple[EyeLandmarkResult, EyeLandmarkResult]:
        uncertain = EyeLandmarkResult("uncertain", None, None)
        return uncertain, uncertain


class ModelSubjectAnalyzer:
    """YuNet faces/landmarks followed by local eye-state inference."""

    def __init__(
        self,
        face_model_path: Path = FACE_MODEL_PATH,
        eye_analyzer: Optional[FaceLandmarkEyeAnalyzer] = None,
    ) -> None:
        self.face_detector = None
        self.error: Optional[str] = None
        try:
            self.face_detector = cv2.FaceDetectorYN.create(
                str(face_model_path),
                "",
                (320, 320),
                0.75,
                0.3,
                5000,
            )
        except cv2.error as exc:
            self.error = str(exc)
        self.eye_analyzer = eye_analyzer or FaceLandmarkEyeAnalyzer()

    @property
    def available(self) -> bool:
        return self.face_detector is not None

    def analyze(self, image: np.ndarray) -> Optional[SubjectModelResult]:
        if self.face_detector is None or image.size == 0:
            return None
        detection, scale = self._detection_image(image)
        height, width = detection.shape[:2]
        try:
            self.face_detector.setInputSize((width, height))
            _status, faces = self.face_detector.detect(detection)
        except cv2.error as exc:
            self.error = str(exc)
            self.face_detector = None
            return None
        model_name = (
            "yunet+face-mesh-litert" if self.eye_analyzer.available else "yunet"
        )
        if faces is None or len(faces) == 0:
            return SubjectModelResult(0, 0, 0, 0, 0, None, None, None, model_name)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        open_count = 0
        closed_count = 0
        uncertain_count = 0
        obscured_count = 0
        face_focus_scores = []
        face_confidences = []
        regions = []
        for face in faces:
            raw_x, raw_y, raw_width, raw_height = [
                int(round(value / scale)) for value in face[:4]
            ]
            x = max(0, raw_x)
            y = max(0, raw_y)
            right = min(gray.shape[1], raw_x + raw_width)
            bottom = min(gray.shape[0], raw_y + raw_height)
            face_width = right - x
            face_height = bottom - y
            if face_width <= 0 or face_height <= 0:
                continue
            face_region = gray[y : y + face_height, x : x + face_width]
            face_focus = None
            if face_region.size:
                face_focus = float(cv2.Laplacian(face_region, cv2.CV_64F).var())
            confidence = float(face[14])
            face_confidences.append(confidence)
            crop_result = self._face_crop_with_bounds(
                image, x, y, face_width, face_height
            )
            eye_results = (
                self.eye_analyzer.analyze(crop_result[0])
                if crop_result is not None
                else FaceLandmarkEyeAnalyzer._uncertain_pair()
            )
            eye_evidence = []
            for eye_result in eye_results:
                normalized_landmarks = ()
                eye_focus = None
                visibility_reason = None
                if crop_result is not None and eye_result.landmarks:
                    normalized_landmarks, pixel_landmarks = self._map_eye_landmarks(
                        eye_result.landmarks,
                        crop_result[1],
                        image.shape[1],
                        image.shape[0],
                    )
                    eye_focus = self._eye_focus_score(gray, pixel_landmarks)
                    visibility_reason = self._eye_visibility_issue(
                        gray, pixel_landmarks, eye_focus
                    )
                elif not eye_result.landmarks:
                    visibility_reason = "Eye landmarks are unavailable"
                state = "obscured" if visibility_reason else eye_result.state
                if state == "open":
                    open_count += 1
                elif state == "closed":
                    closed_count += 1
                elif state == "obscured":
                    obscured_count += 1
                else:
                    uncertain_count += 1
                eye_evidence.append(
                    EyeEvidence(
                        state=state,
                        aperture=eye_result.aperture,
                        openness_score=(
                            None if visibility_reason else eye_result.openness_score
                        ),
                        focus_score=eye_focus,
                        landmarks=normalized_landmarks,
                        visibility_reason=visibility_reason,
                    )
                )
            regions.append(
                FaceEvidence(
                    bounds=(
                        x / image.shape[1],
                        y / image.shape[0],
                        face_width / image.shape[1],
                        face_height / image.shape[0],
                    ),
                    confidence=confidence,
                    focus_score=face_focus,
                    important=False,
                    eyes=tuple(eye_evidence),
                )
            )

        if not regions:
            return SubjectModelResult(0, 0, 0, 0, 0, None, None, None, model_name)
        largest_area = max(region.bounds[2] * region.bounds[3] for region in regions)
        regions = [
            replace(
                region,
                important=region.bounds[2] * region.bounds[3] >= largest_area * 0.25,
            )
            for region in regions
        ]
        important_regions = [region for region in regions if region.important]
        important_eyes = [eye for region in important_regions for eye in region.eyes]
        face_focus_scores = [
            region.focus_score
            for region in important_regions
            if region.focus_score is not None
        ]
        eye_focus_scores = [
            eye.focus_score for eye in important_eyes if eye.focus_score is not None
        ]
        important_open_probabilities = [
            eye.openness_score
            for eye in important_eyes
            if eye.openness_score is not None
        ]

        return SubjectModelResult(
            face_count=len(regions),
            open_eye_count=open_count,
            closed_eye_count=closed_count,
            uncertain_eye_count=uncertain_count,
            obscured_eye_count=obscured_count,
            subject_focus_score=min(face_focus_scores) if face_focus_scores else None,
            face_confidence=(
                sum(face_confidences) / len(face_confidences) if face_confidences else None
            ),
            eye_open_score=(
                sum(important_open_probabilities) / len(important_open_probabilities)
                if important_open_probabilities
                else None
            ),
            model_name=model_name,
            eye_focus_score=min(eye_focus_scores) if eye_focus_scores else None,
            important_face_count=len(important_regions),
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
            regions=tuple(regions),
        )

    @staticmethod
    def _detection_image(image: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= MAX_DETECTION_DIMENSION:
            return image, 1.0
        scale = MAX_DETECTION_DIMENSION / float(longest)
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _face_crop(
        image: np.ndarray, x: int, y: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        result = ModelSubjectAnalyzer._face_crop_with_bounds(
            image, x, y, width, height
        )
        return result[0] if result is not None else None

    @staticmethod
    def _face_crop_with_bounds(
        image: np.ndarray, x: int, y: int, width: int, height: int
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        side = max(width, height) * 1.35
        x1 = max(0, int(round(center_x - side / 2.0)))
        y1 = max(0, int(round(center_y - side / 2.0)))
        x2 = min(image.shape[1], int(round(center_x + side / 2.0)))
        y2 = min(image.shape[0], int(round(center_y + side / 2.0)))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or not math.isfinite(float(crop.mean())):
            return None
        return crop, (x1, y1, x2, y2)

    @staticmethod
    def _map_eye_landmarks(
        landmarks: Tuple[Tuple[float, float], ...],
        crop_bounds: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[Tuple[Tuple[float, float], ...], np.ndarray]:
        x1, y1, x2, y2 = crop_bounds
        crop_width = max(1, x2 - x1)
        crop_height = max(1, y2 - y1)
        pixel_points = np.asarray(
            [
                (
                    min(image_width - 1, max(0.0, x1 + point[0] / 256.0 * crop_width)),
                    min(image_height - 1, max(0.0, y1 + point[1] / 256.0 * crop_height)),
                )
                for point in landmarks
            ],
            dtype=np.float32,
        )
        normalized = tuple(
            (float(point[0] / image_width), float(point[1] / image_height))
            for point in pixel_points
        )
        return normalized, pixel_points

    @staticmethod
    def _eye_focus_score(gray: np.ndarray, landmarks: np.ndarray) -> Optional[float]:
        region = ModelSubjectAnalyzer._eye_region(gray, landmarks)
        if region is None:
            return None
        normalized = cv2.resize(region, (64, 32), interpolation=cv2.INTER_CUBIC)
        return float(cv2.Laplacian(normalized, cv2.CV_64F).var())

    @staticmethod
    def _eye_region(gray: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
        if landmarks.shape != (6, 2) or not np.isfinite(landmarks).all():
            return None
        x, y, width, height = cv2.boundingRect(landmarks.astype(np.float32))
        padding_x = max(2, round(width * 0.35))
        padding_y = max(2, round(max(height, width * 0.22) * 0.55))
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(gray.shape[1], x + width + padding_x)
        y2 = min(gray.shape[0], y + height + padding_y)
        region = gray[y1:y2, x1:x2]
        if region.size < 16 or x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return region

    @staticmethod
    def _eye_visibility_issue(
        gray: np.ndarray,
        landmarks: np.ndarray,
        eye_focus: Optional[float],
    ) -> Optional[str]:
        """Return a conservative visibility issue, never a closed-eye claim."""

        if landmarks.shape != (6, 2) or not np.isfinite(landmarks).all():
            return "Eye landmarks are unavailable"
        eye_width = float(np.linalg.norm(landmarks[0] - landmarks[3]))
        if eye_width < MIN_VISIBLE_EYE_WIDTH:
            return "Eye is too small to verify"
        region = ModelSubjectAnalyzer._eye_region(gray, landmarks)
        if region is None:
            return "Eye region cannot be measured"
        if float(np.count_nonzero(region <= 35) / region.size) >= MAX_DARK_EYE_FRACTION:
            return "Eye region is too dark or covered to verify"
        if float(np.count_nonzero(region >= 235) / region.size) >= MAX_BRIGHT_EYE_FRACTION:
            return "Eye region has strong glare or clipping"
        if float(region.std()) < MIN_VISIBLE_EYE_CONTRAST:
            return "Eye region lacks enough contrast to verify"
        if eye_focus is None or eye_focus < MIN_VISIBLE_EYE_FOCUS:
            return "Eye region lacks enough detail to verify"
        return None
