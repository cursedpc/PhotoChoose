from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from photochoose.database import PhotoDatabase
from photochoose.models import IdentityFaceEvidence

MODEL_DIRECTORY = Path(__file__).with_name("model_assets")
SFACE_MODEL_PATH = MODEL_DIRECTORY / "face_recognition_sface_2021dec.onnx"
IDENTITY_MODEL_NAME = "sface-2021dec-opencv-cpu"

# OpenCV documents a much lower SFace verification threshold. Identity grouping
# deliberately favors precision: a split person can be merged, while a false
# global merge is difficult to notice and contaminates future assignments.
SINGLE_PROTOTYPE_THRESHOLD = 0.58
MULTI_PROTOTYPE_BEST_THRESHOLD = 0.52
MULTI_PROTOTYPE_AGREEMENT_THRESHOLD = 0.48
MINIMUM_WINNER_MARGIN = 0.05
VERY_STRONG_MATCH = 0.72
MINIMUM_IDENTITY_SEED_QUALITY = 0.38
MAX_PROTOTYPES_PER_PERSON = 8


@dataclass(frozen=True)
class IdentityMatchingStats:
    assigned: int = 0
    created_people: int = 0
    unknown: int = 0


class SFaceIdentityEncoder:
    """Align YuNet faces and create normalized SFace identity embeddings."""

    def __init__(self, model_path: Path = SFACE_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._recognizer = None
        self.error: Optional[str] = None
        if not self.model_path.is_file():
            self.error = "SFace model is unavailable"

    def _activate(self) -> bool:
        if self._recognizer is not None:
            return True
        if not self.model_path.is_file():
            return False
        try:
            creator = getattr(cv2, "FaceRecognizerSF_create", None)
            if creator is not None:
                self._recognizer = creator(str(self.model_path), "")
            else:
                self._recognizer = cv2.FaceRecognizerSF.create(
                    str(self.model_path), ""
                )
        except (AttributeError, cv2.error) as exc:
            self.error = str(exc)
            self._recognizer = None
        return self._recognizer is not None

    @property
    def available(self) -> bool:
        return self.model_path.is_file() and self.error is None

    def encode(
        self,
        image: np.ndarray,
        detected_face: np.ndarray,
        face_index: int,
        bounds: Tuple[float, float, float, float],
        focus_score: Optional[float],
    ) -> Optional[IdentityFaceEvidence]:
        if image.size == 0 or not self._activate() or self._recognizer is None:
            return None
        face = np.asarray(detected_face, dtype=np.float32).reshape(-1)
        if face.size < 14 or not np.isfinite(face[:14]).all():
            return None
        try:
            aligned = self._recognizer.alignCrop(image, face)
            feature = self._recognizer.feature(aligned)
        except (cv2.error, RuntimeError, ValueError) as exc:
            self.error = str(exc)
            return None
        embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not embedding.size or not np.isfinite(embedding).all() or norm <= 1e-8:
            return None
        embedding /= norm
        landmarks = tuple(
            (
                float(face[position] / image.shape[1]),
                float(face[position + 1] / image.shape[0]),
            )
            for position in range(4, 14, 2)
        )
        return IdentityFaceEvidence(
            face_index=face_index,
            bounds=bounds,
            landmarks=landmarks,
            embedding=tuple(float(value) for value in embedding),
            quality=self._quality(bounds, float(face[14]) if face.size > 14 else 1.0, focus_score),
            model_name=IDENTITY_MODEL_NAME,
        )

    @staticmethod
    def _quality(
        bounds: Sequence[float], confidence: float, focus_score: Optional[float]
    ) -> float:
        area = max(0.0, float(bounds[2]) * float(bounds[3]))
        size_quality = min(1.0, np.sqrt(area) / 0.22)
        focus_quality = 0.45 if focus_score is None else min(1.0, focus_score / 180.0)
        return float(
            max(
                0.0,
                min(1.0, 0.45 * confidence + 0.35 * size_quality + 0.20 * focus_quality),
            )
        )


class IdentityMatcher:
    """Incrementally assign local face embeddings to persistent global people."""

    def __init__(self, database: PhotoDatabase) -> None:
        self.database = database

    def assign_unknown_faces(self) -> IdentityMatchingStats:
        faces = self.database.list_identity_faces()
        prototypes = self._prototype_map(faces)
        unknown = sorted(
            (
                face
                for face in faces
                if face["person_id"] is None
                and face["assignment_source"] != "manual"
            ),
            key=lambda face: (-float(face["quality"]), int(face["id"])),
        )
        assigned = 0
        created = 0
        unresolved = 0
        for face in unknown:
            person_id, confidence = self._best_person(face, prototypes)
            if person_id is None:
                if float(face["quality"]) < MINIMUM_IDENTITY_SEED_QUALITY:
                    unresolved += 1
                    continue
                person_id = self.database.create_person()
                confidence = 1.0
                created += 1
            self.database.assign_identity_face(
                int(face["id"]), person_id, confidence, source="automatic"
            )
            prototype = dict(face)
            prototype["person_id"] = person_id
            prototypes.setdefault(person_id, []).append(prototype)
            prototypes[person_id] = self._select_prototypes(prototypes[person_id])
            assigned += 1
        return IdentityMatchingStats(assigned, created, unresolved)

    def _best_person(
        self,
        face: Dict[str, object],
        prototypes: Dict[int, List[Dict[str, object]]],
    ) -> Tuple[Optional[int], float]:
        rejected = set(self.database.rejected_person_ids(int(face["id"])))
        candidates = []
        for person_id, examples in prototypes.items():
            if person_id in rejected or self.database.person_has_photo(
                person_id, int(face["photo_id"]), excluding_face_id=int(face["id"])
            ):
                continue
            similarities = sorted(
                (
                    float(np.dot(face["embedding"], example["embedding"]))
                    for example in examples
                    if np.asarray(example["embedding"]).shape
                    == np.asarray(face["embedding"]).shape
                ),
                reverse=True,
            )
            if not similarities:
                continue
            named_identity = any(
                bool(example.get("person_is_named")) for example in examples
            )
            best = similarities[0]
            agreement = (
                best
                if len(similarities) == 1
                else float(sum(similarities[:2]) / 2.0)
            )
            if len(similarities) == 1:
                accepted = best >= SINGLE_PROTOTYPE_THRESHOLD + (
                    0.04 if named_identity else 0.0
                )
            else:
                accepted = (
                    best
                    >= MULTI_PROTOTYPE_BEST_THRESHOLD
                    + (0.04 if named_identity else 0.0)
                    and agreement
                    >= MULTI_PROTOTYPE_AGREEMENT_THRESHOLD
                    + (0.03 if named_identity else 0.0)
                )
            if accepted:
                candidates.append((agreement, best, person_id))
        if not candidates:
            return None, 0.0
        candidates.sort(reverse=True)
        agreement, best, person_id = candidates[0]
        runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
        required_margin = MINIMUM_WINNER_MARGIN + (
            0.02
            if any(
                bool(example.get("person_is_named"))
                for example in prototypes[person_id]
            )
            else 0.0
        )
        if best < VERY_STRONG_MATCH and agreement - runner_up < required_margin:
            return None, 0.0
        return person_id, max(0.0, min(1.0, agreement))

    @classmethod
    def _prototype_map(
        cls, faces: Iterable[Dict[str, object]]
    ) -> Dict[int, List[Dict[str, object]]]:
        grouped: Dict[int, List[Dict[str, object]]] = {}
        for face in faces:
            person_id = face["person_id"]
            if person_id is None:
                continue
            grouped.setdefault(int(person_id), []).append(face)
        return {
            person_id: cls._select_prototypes(examples)
            for person_id, examples in grouped.items()
        }

    @staticmethod
    def _select_prototypes(
        faces: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        # Limit near-identical burst frames by keeping at most two examples per
        # folder before filling the remaining slots with the best other faces.
        ranked = sorted(
            faces,
            key=lambda face: (-float(face["quality"]), int(face["id"])),
        )
        chosen: List[Dict[str, object]] = []
        per_project: Dict[int, int] = {}
        for face in ranked:
            project_id = int(face["project_id"])
            if per_project.get(project_id, 0) >= 2:
                continue
            chosen.append(face)
            per_project[project_id] = per_project.get(project_id, 0) + 1
            if len(chosen) >= MAX_PROTOTYPES_PER_PERSON:
                return chosen
        chosen_ids = {int(face["id"]) for face in chosen}
        for face in ranked:
            if int(face["id"]) in chosen_ids:
                continue
            chosen.append(face)
            if len(chosen) >= MAX_PROTOTYPES_PER_PERSON:
                break
        return chosen
