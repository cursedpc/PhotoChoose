from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from photochoose.inference_backend import (
    InferenceBackendError,
    OpenVINOCPUBackend,
    default_openvino_cpu_backend,
)
from photochoose.models import ObjectEvidence

MODEL_DIRECTORY = Path(__file__).with_name("model_assets")
OBJECT_MODEL_PATH = MODEL_DIRECTORY / "object_detection_nanodet_2022nov.onnx"
OBJECT_INPUT_SIZE = (416, 416)
OBJECT_CONFIDENCE_THRESHOLD = 0.35
OBJECT_EVIDENCE_CONFIDENCE = 0.42
PERSON_EVIDENCE_CONFIDENCE = 0.36
OBJECT_NMS_THRESHOLD = 0.60
OBJECT_STRIDES = (8, 16, 32)
OBJECT_REG_MAX = 7
OBJECT_OUTPUT_NAMES = ("792", "795", "814", "817", "836", "839")

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class GeneralSubjectResult:
    object_count: int
    primary_focus_score: Optional[float]
    model_name: str
    objects: Tuple[ObjectEvidence, ...]


class GeneralSubjectAnalyzer:
    """Compact local object detection with a model-free saliency fallback."""

    def __init__(
        self,
        model_path: Path = OBJECT_MODEL_PATH,
        inference: Optional[Callable[[np.ndarray], Sequence[np.ndarray]]] = None,
        backend: Optional[OpenVINOCPUBackend] = None,
    ) -> None:
        self._inference = inference
        self._model_path = model_path
        self.backend_name = "custom" if inference is not None else "none"
        self.fallback_reason: Optional[str] = None
        self.error: Optional[str] = None
        if self._inference is None:
            backend = backend or default_openvino_cpu_backend()
            if backend is not None:
                try:
                    model = backend.compile_model(model_path)
                    self._inference = lambda blob: model.infer(
                        blob, OBJECT_OUTPUT_NAMES
                    )
                    self.backend_name = backend.name
                except InferenceBackendError as exc:
                    self.fallback_reason = str(exc)
            if self._inference is None:
                self._activate_opencv()
        self._anchors = tuple(
            self._anchors_for_stride(stride) for stride in OBJECT_STRIDES
        )
        self._project = np.arange(OBJECT_REG_MAX + 1, dtype=np.float32)

    def _activate_opencv(self) -> bool:
        try:
            network = cv2.dnn.readNet(str(self._model_path))
            output_names = network.getUnconnectedOutLayersNames()

            def infer(blob: np.ndarray) -> Sequence[np.ndarray]:
                network.setInput(blob)
                return network.forward(output_names)

            self._inference = infer
            self.backend_name = "opencv"
            self.error = None
            return True
        except cv2.error as exc:
            self.error = str(exc)
            self._inference = None
            self.backend_name = "none"
            return False

    def _run_inference(self, blob: np.ndarray) -> Sequence[np.ndarray]:
        if self._inference is None:
            raise RuntimeError("Object inference is unavailable")
        try:
            return self._inference(blob)
        except (InferenceBackendError, RuntimeError, ValueError) as exc:
            if self.backend_name != "openvino-cpu":
                raise
            self.fallback_reason = str(exc)
            self._inference = None
            if not self._activate_opencv() or self._inference is None:
                raise
            return self._inference(blob)

    @property
    def available(self) -> bool:
        return self._inference is not None

    def analyze(self, image: np.ndarray) -> GeneralSubjectResult:
        if image.size == 0:
            return GeneralSubjectResult(0, None, "none", ())
        objects: Tuple[ObjectEvidence, ...] = ()
        model_name = "spectral-saliency"
        if self._inference is not None:
            try:
                prepared, letterbox = self._letterbox(image)
                blob = self._preprocess(prepared)
                outputs = self._run_inference(blob)
                boxes = self._postprocess(outputs)
                objects = self._object_evidence(image, boxes, letterbox)
                model_name = "nanodet-{0}".format(self.backend_name)
            except (cv2.error, RuntimeError, ValueError, InferenceBackendError) as exc:
                self.error = str(exc)
                self._inference = None
                self.backend_name = "none"
        if not objects:
            salient = self._salient_object(image)
            objects = (salient,) if salient is not None else ()
            if model_name.startswith("nanodet-") and objects:
                model_name += "+spectral-saliency"
        primary_scores = [
            item.focus_score
            for item in objects
            if item.importance == "primary" and item.focus_score is not None
        ]
        return GeneralSubjectResult(
            object_count=len(objects),
            primary_focus_score=min(primary_scores) if primary_scores else None,
            model_name=model_name,
            objects=objects,
        )

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        mean = np.asarray((103.53, 116.28, 123.675), dtype=np.float32)
        std = np.asarray((57.375, 57.12, 58.395), dtype=np.float32)
        return cv2.dnn.blobFromImage((rgb - mean) / std)

    @staticmethod
    def _letterbox(
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        target_height, target_width = OBJECT_INPUT_SIZE
        height, width = image.shape[:2]
        scale = min(target_width / float(width), target_height / float(height))
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )
        left = (target_width - new_width) // 2
        top = (target_height - new_height) // 2
        prepared = cv2.copyMakeBorder(
            resized,
            top,
            target_height - new_height - top,
            left,
            target_width - new_width - left,
            cv2.BORDER_CONSTANT,
            value=0,
        )
        return prepared, (top, left, new_height, new_width)

    def _postprocess(self, outputs: Sequence[np.ndarray]) -> np.ndarray:
        if len(outputs) < len(OBJECT_STRIDES) * 2:
            raise ValueError("NanoDet returned an incomplete output set")
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        class_outputs = outputs[::2]
        box_outputs = outputs[1::2]
        for stride, anchors, class_output, box_output in zip(
            OBJECT_STRIDES, self._anchors, class_outputs, box_outputs
        ):
            class_scores = np.asarray(class_output, dtype=np.float32).reshape(-1, 80)
            distributions = np.asarray(box_output, dtype=np.float32).reshape(
                -1, OBJECT_REG_MAX + 1
            )
            distributions -= distributions.max(axis=1, keepdims=True)
            probabilities = np.exp(distributions)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            distances = np.dot(probabilities, self._project).reshape(-1, 4) * stride
            if len(anchors) != len(class_scores):
                raise ValueError("NanoDet output shape does not match its anchors")
            if len(class_scores) > 1000:
                top = np.argsort(class_scores.max(axis=1))[::-1][:1000]
                class_scores = class_scores[top]
                distances = distances[top]
                points = anchors[top]
            else:
                points = anchors
            boxes = np.column_stack(
                (
                    points[:, 0] - distances[:, 0],
                    points[:, 1] - distances[:, 1],
                    points[:, 0] + distances[:, 2],
                    points[:, 1] + distances[:, 3],
                )
            )
            np.clip(boxes, 0.0, float(OBJECT_INPUT_SIZE[0]), out=boxes)
            all_boxes.append(boxes)
            all_scores.append(class_scores)
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        widths_heights = boxes.copy()
        widths_heights[:, 2:] -= widths_heights[:, :2]
        indices = cv2.dnn.NMSBoxes(
            widths_heights.tolist(),
            confidences.tolist(),
            OBJECT_CONFIDENCE_THRESHOLD,
            OBJECT_NMS_THRESHOLD,
        )
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)
        selected = np.asarray(indices, dtype=np.int32).reshape(-1)
        return np.column_stack(
            (boxes[selected], confidences[selected], class_ids[selected])
        ).astype(np.float32)

    def _object_evidence(
        self,
        image: np.ndarray,
        detections: np.ndarray,
        letterbox: Tuple[int, int, int, int],
    ) -> Tuple[ObjectEvidence, ...]:
        height, width = image.shape[:2]
        top, left, new_height, new_width = letterbox
        items: List[Tuple[ObjectEvidence, float]] = []
        for detection in detections:
            x1 = (float(detection[0]) - left) * width / new_width
            y1 = (float(detection[1]) - top) * height / new_height
            x2 = (float(detection[2]) - left) * width / new_width
            y2 = (float(detection[3]) - top) * height / new_height
            x1 = min(float(width), max(0.0, x1))
            y1 = min(float(height), max(0.0, y1))
            x2 = min(float(width), max(0.0, x2))
            y2 = min(float(height), max(0.0, y2))
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width < 4.0 or box_height < 4.0:
                continue
            normalized = (x1 / width, y1 / height, box_width / width, box_height / height)
            confidence = float(detection[4])
            class_id = int(detection[5])
            minimum_confidence = (
                PERSON_EVIDENCE_CONFIDENCE
                if COCO_CLASSES[class_id] == "person"
                else OBJECT_EVIDENCE_CONFIDENCE
            )
            if confidence < minimum_confidence:
                continue
            focus = self._region_focus(image, normalized)
            area = normalized[2] * normalized[3]
            center_x = normalized[0] + normalized[2] / 2.0
            center_y = normalized[1] + normalized[3] / 2.0
            center_distance = math.hypot(center_x - 0.5, center_y - 0.5) / math.sqrt(0.5)
            salience = confidence * math.sqrt(area) * (1.0 - 0.30 * min(1.0, center_distance))
            items.append(
                (
                    ObjectEvidence(
                        label=COCO_CLASSES[class_id],
                        bounds=normalized,
                        confidence=confidence,
                        focus_score=focus,
                        importance="background",
                    ),
                    salience,
                )
            )
        if not items:
            return ()
        maximum = max(score for _item, score in items)
        ranked = sorted(items, key=lambda pair: pair[1], reverse=True)
        classified = []
        for index, (item, score) in enumerate(ranked):
            area = item.bounds[2] * item.bounds[3]
            primary_candidate = area >= 0.012 or (
                item.confidence >= 0.60 and area >= 0.004
            )
            if primary_candidate and (index == 0 or score >= maximum * 0.55):
                importance = "primary"
            elif score >= maximum * 0.20 or area >= 0.04:
                importance = "supporting"
            else:
                importance = "background"
            classified.append(replace(item, importance=importance))
        return tuple(classified)

    def _salient_object(self, image: np.ndarray) -> Optional[ObjectEvidence]:
        height, width = image.shape[:2]
        scale = min(1.0, 256.0 / max(height, width))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        spectrum = np.fft.fft2(small / 255.0)
        amplitude = np.log(np.abs(spectrum) + 1e-8)
        residual = amplitude - cv2.blur(amplitude, (3, 3))
        reconstructed = np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))
        saliency = np.abs(reconstructed) ** 2
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (9, 9), 0)
        maximum = float(saliency.max())
        if maximum <= 1e-8:
            return None
        saliency /= maximum
        threshold = max(0.20, float(np.percentile(saliency, 85.0)))
        mask = np.uint8(saliency >= threshold) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = []
        total_area = float(mask.shape[0] * mask.shape[1])
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = box_width * box_height / total_area
            if area < 0.01 or area > 0.85:
                continue
            component = saliency[y : y + box_height, x : x + box_width]
            strength = float(component.mean()) * math.sqrt(area)
            center_x = (x + box_width / 2.0) / mask.shape[1]
            center_y = (y + box_height / 2.0) / mask.shape[0]
            distance = math.hypot(center_x - 0.5, center_y - 0.5) / math.sqrt(0.5)
            candidates.append(
                (
                    strength * (1.0 - 0.25 * min(1.0, distance)),
                    x,
                    y,
                    box_width,
                    box_height,
                )
            )
        if not candidates:
            return None
        _score, x, y, box_width, box_height = max(candidates)
        padding_x = max(2, round(box_width * 0.10))
        padding_y = max(2, round(box_height * 0.10))
        x1 = max(0, x - padding_x) / scale
        y1 = max(0, y - padding_y) / scale
        x2 = min(mask.shape[1], x + box_width + padding_x) / scale
        y2 = min(mask.shape[0], y + box_height + padding_y) / scale
        bounds = (
            x1 / width,
            y1 / height,
            min(width, x2) / width - x1 / width,
            min(height, y2) / height - y1 / height,
        )
        return ObjectEvidence(
            label="prominent region",
            bounds=bounds,
            confidence=0.25,
            focus_score=self._region_focus(image, bounds),
            importance="primary",
        )

    @staticmethod
    def _region_focus(
        image: np.ndarray, bounds: Tuple[float, float, float, float]
    ) -> Optional[float]:
        height, width = image.shape[:2]
        x = max(0, min(width - 1, round(bounds[0] * width)))
        y = max(0, min(height - 1, round(bounds[1] * height)))
        right = max(x + 1, min(width, round((bounds[0] + bounds[2]) * width)))
        bottom = max(y + 1, min(height, round((bounds[1] + bounds[3]) * height)))
        region = image[y:bottom, x:right]
        if region.size == 0 or min(region.shape[:2]) < 8:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        longest = max(gray.shape)
        if longest > 256:
            scale = 256.0 / longest
            gray = cv2.resize(
                gray,
                (max(8, round(gray.shape[1] * scale)), max(8, round(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _anchors_for_stride(stride: int) -> np.ndarray:
        feature_width = OBJECT_INPUT_SIZE[1] // stride
        feature_height = OBJECT_INPUT_SIZE[0] // stride
        x_values, y_values = np.meshgrid(
            np.arange(feature_width, dtype=np.float32) * stride,
            np.arange(feature_height, dtype=np.float32) * stride,
        )
        return np.column_stack(
            (
                x_values.reshape(-1) + 0.5 * (stride - 1),
                y_values.reshape(-1) + 0.5 * (stride - 1),
            )
        )
