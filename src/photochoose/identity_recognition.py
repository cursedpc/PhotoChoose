from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from photochoose.database import PhotoDatabase
from photochoose.inference_backend import (
    InferenceBackendError,
    OpenVINOCPUBackend,
    default_openvino_cpu_backend,
)
from photochoose.models import IdentityFaceEvidence

MODEL_DIRECTORY = Path(__file__).with_name("model_assets")
SFACE_MODEL_PATH = MODEL_DIRECTORY / "face_recognition_sface_2021dec.onnx"
OPENVINO_REID_MODEL_PATH = (
    MODEL_DIRECTORY / "face-reidentification-retail-0095.xml"
)
SFACE_MODEL_NAME = "sface-2021dec-opencv-cpu"
OPENVINO_REID_MODEL_NAME = "face-reidentification-retail-0095-openvino-cpu"
IDENTITY_MODEL_NAME = OPENVINO_REID_MODEL_NAME
IDENTITY_GROUPING_VERSION = "identity-groups-v7-openvino-consolidated-album"

OPENVINO_REID_REFERENCE_LANDMARKS = np.asarray(
    (
        (0.31556875, 0.4615741071428571),
        (0.6826229166666667, 0.4615741071428571),
        (0.5002625, 0.6405053571428571),
        (0.349471875, 0.8246919642857142),
        (0.6534364583333333, 0.8246919642857142),
    ),
    dtype=np.float32,
) * 128.0

# OpenCV documents a much lower SFace verification threshold. Identity grouping
# deliberately favors precision: a split person can be merged, while a false
# global merge is difficult to notice and contaminates future assignments.
SINGLE_PROTOTYPE_THRESHOLD = 0.82
MULTI_PROTOTYPE_BEST_THRESHOLD = 0.78
MULTI_PROTOTYPE_AGREEMENT_THRESHOLD = 0.72
MINIMUM_WINNER_MARGIN = 0.05
VERY_STRONG_MATCH = 0.86
MINIMUM_IDENTITY_SEED_QUALITY = 0.52
MINIMUM_IDENTITY_SEED_AREA = 0.008
MINIMUM_NEW_PERSON_PHOTOS = 3
NEW_PERSON_COMPLETE_LINK_THRESHOLD = 0.64
NEW_PERSON_STRONG_PEER_THRESHOLD = 0.76
NEW_PERSON_MIN_STRONG_PEERS = 2
SEED_CLUSTER_MERGE_THRESHOLD = 0.75
LARGE_CLUSTER_MERGE_THRESHOLD = 0.68
LARGE_CLUSTER_MIN_PHOTOS = 20
LARGE_CLUSTER_MEMBER_SUPPORT = 0.60
RECURRING_PERSON_MIN_PHOTOS = 12
RECURRING_PERSON_MIN_COVERAGE = 0.30
RECURRING_PERSON_CENTROID_THRESHOLD = 0.50
RECURRING_PERSON_MARGIN = 0.06
MAX_PROTOTYPES_PER_PERSON = 8
DOMINANT_PERSON_MIN_COVERAGE = 0.65
DOMINANT_PERSON_MEMBER_AGREEMENT = 0.35
DOMINANT_PERSON_FACE_MARGIN = 0.08
AUTOMATIC_CLUSTER_MERGE_MARGIN = 0.06


@dataclass(frozen=True)
class SimilarityThresholds:
    single_prototype: float
    multi_prototype_best: float
    multi_prototype_agreement: float
    minimum_winner_margin: float
    very_strong: float
    new_person_complete_link: float
    new_person_strong_peer: float
    seed_cluster_merge: float
    large_cluster_merge: float
    large_cluster_member_support: float
    recurring_centroid: float
    recurring_margin: float


DEFAULT_THRESHOLDS = SimilarityThresholds(
    SINGLE_PROTOTYPE_THRESHOLD,
    MULTI_PROTOTYPE_BEST_THRESHOLD,
    MULTI_PROTOTYPE_AGREEMENT_THRESHOLD,
    MINIMUM_WINNER_MARGIN,
    VERY_STRONG_MATCH,
    NEW_PERSON_COMPLETE_LINK_THRESHOLD,
    NEW_PERSON_STRONG_PEER_THRESHOLD,
    SEED_CLUSTER_MERGE_THRESHOLD,
    LARGE_CLUSTER_MERGE_THRESHOLD,
    LARGE_CLUSTER_MEMBER_SUPPORT,
    RECURRING_PERSON_CENTROID_THRESHOLD,
    RECURRING_PERSON_MARGIN,
)

# These cosine thresholds were measured on the photographer's 286-photo test
# folder. Co-occurring faces provide hard negative evidence, while the recurring
# pass can bridge pose and quality changes only after a strong album-level core
# is established.
OPENVINO_REID_THRESHOLDS = SimilarityThresholds(
    single_prototype=0.66,
    multi_prototype_best=0.60,
    multi_prototype_agreement=0.52,
    minimum_winner_margin=0.08,
    very_strong=0.76,
    new_person_complete_link=0.42,
    new_person_strong_peer=0.56,
    seed_cluster_merge=0.58,
    large_cluster_merge=0.45,
    large_cluster_member_support=0.44,
    recurring_centroid=0.38,
    recurring_margin=0.09,
)


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
            model_name=SFACE_MODEL_NAME,
        )

    @staticmethod
    def _quality(
        bounds: Sequence[float], confidence: float, focus_score: Optional[float]
    ) -> float:
        area = max(0.0, float(bounds[2]) * float(bounds[3]))
        size_quality = float(np.clip((np.sqrt(area) - 0.055) / 0.18, 0.0, 1.0))
        focus_quality = 0.45 if focus_score is None else min(1.0, focus_score / 180.0)
        return float(
            max(
                0.0,
                min(
                    1.0,
                    0.25 * confidence + 0.50 * size_quality + 0.25 * focus_quality,
                ),
            )
        )


class OpenVINOIdentityEncoder:
    """Create cosine-comparable 256-value OpenVINO face embeddings."""

    def __init__(
        self,
        model_path: Path = OPENVINO_REID_MODEL_PATH,
        backend: Optional[OpenVINOCPUBackend] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self._backend = backend
        self._model = None
        self._output_name: Optional[str] = None
        self.error: Optional[str] = None
        if (
            not self.model_path.is_file()
            or not self.model_path.with_suffix(".bin").is_file()
        ):
            self.error = "OpenVINO face re-identification model is unavailable"

    def _activate(self) -> bool:
        if self._model is not None:
            return True
        if (
            not self.model_path.is_file()
            or not self.model_path.with_suffix(".bin").is_file()
        ):
            return False
        backend = self._backend or default_openvino_cpu_backend()
        if backend is None:
            self.error = "OpenVINO CPU backend is unavailable"
            return False
        try:
            self._model = backend.compile_model(self.model_path, (1, 3, 128, 128))
            self._output_name = self._model.output_names[0]
            self.error = None
        except (IndexError, InferenceBackendError) as exc:
            self.error = str(exc)
            self._model = None
            self._output_name = None
        return self._model is not None

    @property
    def available(self) -> bool:
        return (
            self.model_path.is_file()
            and self.model_path.with_suffix(".bin").is_file()
            and self.error is None
        )

    def encode(
        self,
        image: np.ndarray,
        detected_face: np.ndarray,
        face_index: int,
        bounds: Tuple[float, float, float, float],
        focus_score: Optional[float],
    ) -> Optional[IdentityFaceEvidence]:
        if (
            image.size == 0
            or not self._activate()
            or self._model is None
            or self._output_name is None
        ):
            return None
        face = np.asarray(detected_face, dtype=np.float32).reshape(-1)
        if face.size < 14 or not np.isfinite(face[:14]).all():
            return None
        landmarks = face[4:14].reshape(5, 2)
        transform, _inliers = cv2.estimateAffinePartial2D(
            landmarks,
            OPENVINO_REID_REFERENCE_LANDMARKS,
            method=cv2.LMEDS,
        )
        if transform is None or not np.isfinite(transform).all():
            return None
        aligned = cv2.warpAffine(
            image,
            transform,
            (128, 128),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tensor = aligned.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        try:
            feature = self._model.infer(tensor, (self._output_name,))[0]
        except InferenceBackendError as exc:
            self.error = str(exc)
            return None
        embedding = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.size != 256 or not np.isfinite(embedding).all() or norm <= 1e-8:
            return None
        embedding /= norm
        normalized_landmarks = tuple(
            (
                float(point[0] / image.shape[1]),
                float(point[1] / image.shape[0]),
            )
            for point in landmarks
        )
        return IdentityFaceEvidence(
            face_index=face_index,
            bounds=bounds,
            landmarks=normalized_landmarks,
            embedding=tuple(float(value) for value in embedding),
            quality=SFaceIdentityEncoder._quality(
                bounds,
                float(face[14]) if face.size > 14 else 1.0,
                focus_score,
            ),
            model_name=OPENVINO_REID_MODEL_NAME,
        )


class StrongIdentityEncoder:
    """Prefer OpenVINO re-identification and retain SFace as recovery."""

    def __init__(
        self,
        reidentification: Optional[OpenVINOIdentityEncoder] = None,
        sface: Optional[SFaceIdentityEncoder] = None,
    ) -> None:
        self.reidentification = reidentification or OpenVINOIdentityEncoder()
        self.sface = sface or SFaceIdentityEncoder()

    @property
    def available(self) -> bool:
        return self.reidentification.available or self.sface.available

    @property
    def error(self) -> Optional[str]:
        return (
            self.reidentification.error
            if self.reidentification.available
            else self.sface.error
        )

    def encode(
        self,
        image: np.ndarray,
        detected_face: np.ndarray,
        face_index: int,
        bounds: Tuple[float, float, float, float],
        focus_score: Optional[float],
    ) -> Optional[IdentityFaceEvidence]:
        if self.reidentification._activate():
            return self.reidentification.encode(
                image, detected_face, face_index, bounds, focus_score
            )
        return self.sface.encode(
            image, detected_face, face_index, bounds, focus_score
        )


class IdentityMatcher:
    """Incrementally assign local face embeddings to persistent global people."""

    def __init__(self, database: PhotoDatabase) -> None:
        self.database = database

    @staticmethod
    def _thresholds_for(face: Dict[str, object]) -> SimilarityThresholds:
        if str(face.get("model_name") or "") == OPENVINO_REID_MODEL_NAME:
            return OPENVINO_REID_THRESHOLDS
        return DEFAULT_THRESHOLDS

    def assign_unknown_faces(self) -> IdentityMatchingStats:
        self.database.prepare_identity_regroup(IDENTITY_GROUPING_VERSION)
        faces = self.database.list_identity_faces()
        prototypes = self._prototype_map(faces)
        members = self._member_map(faces)
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
        remaining = []

        # First attach strong matches to identities that are already trusted,
        # especially user-named people retained across grouping revisions.
        for face in unknown:
            person_id, confidence = self._best_person(face, prototypes)
            if person_id is None:
                remaining.append(face)
                continue
            self._assign(face, person_id, confidence, prototypes, members)
            assigned += 1

        # A new anonymous person requires consistent evidence across at least
        # three different photographs. Single faces and pairs remain Unknown.
        discovered_ids = set()
        for cluster in self._discover_new_people(remaining):
            person_id = self.database.create_person()
            confidence = self._cluster_confidence(cluster)
            for face in cluster:
                self._assign(face, person_id, confidence, prototypes, members)
                discovered_ids.add(int(face["id"]))
                assigned += 1
            created += 1

        # Newly created multi-photo identities may safely absorb additional
        # weaker faces, but those weak faces can never seed a person themselves.
        for face in remaining:
            if int(face["id"]) in discovered_ids:
                continue
            person_id, confidence = self._best_person(face, prototypes)
            if person_id is None:
                continue
            self._assign(face, person_id, confidence, prototypes, members)
            discovered_ids.add(int(face["id"]))
            assigned += 1

        # A strict first pass may create a small appearance fragment beside a
        # larger identity (profile views beside frontal views, for example).
        # Consolidate only whole, automatic groups backed by aggregate evidence;
        # faces that co-occur in one photograph can never be the same person.
        self._consolidate_automatic_people(prototypes, members)

        recurring_assigned = self._assign_recurring_faces(
            [face for face in remaining if int(face["id"]) not in discovered_ids],
            prototypes,
            members,
            faces,
        )
        assigned += recurring_assigned

        self.database.finish_identity_regroup(IDENTITY_GROUPING_VERSION)
        return IdentityMatchingStats(assigned, created, len(unknown) - assigned)

    def _consolidate_automatic_people(
        self,
        prototypes: Dict[int, List[Dict[str, object]]],
        members: Dict[int, List[Dict[str, object]]],
    ) -> int:
        """Merge conservatively split automatic identities using all members."""

        merged_count = 0
        while True:
            candidates = []
            person_ids = sorted(members)
            for left_index, left_id in enumerate(person_ids):
                left = members[left_id]
                if not left:
                    continue
                for right_id in person_ids[left_index + 1 :]:
                    right = members[right_id]
                    if not right or not self._automatic_merge_allowed(left, right):
                        continue
                    compatible_scores = []
                    for left_group in self._compatible_groups(left).values():
                        left_key = self._embedding_key(left_group[0])
                        right_group = self._compatible_groups(right).get(left_key)
                        if not right_group:
                            continue
                        score = float(
                            np.dot(
                                self._cluster_centroid(left_group),
                                self._cluster_centroid(right_group),
                            )
                        )
                        thresholds = self._thresholds_for(left_group[0])
                        larger, smaller = (
                            (left_group, right_group)
                            if len(left_group) >= len(right_group)
                            else (right_group, left_group)
                        )
                        minimum = thresholds.seed_cluster_merge
                        if self._cluster_supported_by(
                            smaller,
                            larger,
                            thresholds.large_cluster_member_support,
                        ):
                            minimum = thresholds.large_cluster_merge + 0.08
                        if score >= minimum:
                            compatible_scores.append(score)
                    if compatible_scores:
                        candidates.append((max(compatible_scores), left_id, right_id))
            if not candidates:
                return merged_count

            candidates.sort(reverse=True)
            best_score, left_id, right_id = candidates[0]
            runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
            if (
                best_score < self._thresholds_for(members[left_id][0]).very_strong
                and best_score - runner_up < AUTOMATIC_CLUSTER_MERGE_MARGIN
            ):
                return merged_count

            left_named = any(
                bool(face.get("person_is_named")) for face in members[left_id]
            )
            right_named = any(
                bool(face.get("person_is_named")) for face in members[right_id]
            )
            left_protected = any(
                face.get("assignment_source") != "automatic"
                for face in members[left_id]
            )
            right_protected = any(
                face.get("assignment_source") != "automatic"
                for face in members[right_id]
            )
            if left_named:
                target_id, source_id = left_id, right_id
            elif right_named:
                target_id, source_id = right_id, left_id
            elif left_protected:
                target_id, source_id = left_id, right_id
            elif right_protected:
                target_id, source_id = right_id, left_id
            elif len(members[left_id]) >= len(members[right_id]):
                target_id, source_id = left_id, right_id
            else:
                target_id, source_id = right_id, left_id
            self.database.merge_automatic_person(target_id, source_id, best_score)
            moved = members.pop(source_id)
            target_named = any(
                bool(face.get("person_is_named")) for face in members[target_id]
            )
            for face in moved:
                face["person_id"] = target_id
                face["person_is_named"] = target_named
            members[target_id].extend(moved)
            prototypes.pop(source_id, None)
            prototypes[target_id] = self._select_compatible_prototypes(
                members[target_id]
            )
            merged_count += 1

    @classmethod
    def _automatic_merge_allowed(
        cls,
        left: Sequence[Dict[str, object]],
        right: Sequence[Dict[str, object]],
    ) -> bool:
        left_protected = any(
            face.get("assignment_source") != "automatic" for face in left
        )
        right_protected = any(
            face.get("assignment_source") != "automatic" for face in right
        )
        left_named = any(bool(face.get("person_is_named")) for face in left)
        right_named = any(bool(face.get("person_is_named")) for face in right)
        if left_named and right_named:
            return False
        if left_named and right_protected:
            return False
        if right_named and left_protected:
            return False
        if left_protected and right_protected:
            return False
        return not {
            int(face["photo_id"]) for face in left
        }.intersection(int(face["photo_id"]) for face in right)

    @staticmethod
    def _embedding_key(face: Dict[str, object]) -> Tuple[str, int]:
        embedding = np.asarray(face["embedding"])
        return str(face.get("model_name") or ""), int(embedding.size)

    @classmethod
    def _compatible_groups(
        cls, faces: Sequence[Dict[str, object]]
    ) -> Dict[Tuple[str, int], List[Dict[str, object]]]:
        grouped: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
        for face in faces:
            grouped.setdefault(cls._embedding_key(face), []).append(face)
        return grouped

    def _assign(
        self,
        face: Dict[str, object],
        person_id: int,
        confidence: float,
        prototypes: Dict[int, List[Dict[str, object]]],
        members: Dict[int, List[Dict[str, object]]],
    ) -> None:
        self.database.assign_identity_face(
            int(face["id"]), person_id, confidence, source="automatic"
        )
        prototype = dict(face)
        prototype["person_id"] = person_id
        prototype["assignment_source"] = "automatic"
        prototype["assignment_confidence"] = confidence
        prototype["person_is_named"] = any(
            bool(item.get("person_is_named"))
            for item in prototypes.get(person_id, ())
        )
        members.setdefault(person_id, []).append(prototype)
        prototypes.setdefault(person_id, []).append(prototype)
        prototypes[person_id] = self._select_compatible_prototypes(
            prototypes[person_id]
        )

    def _discover_new_people(
        self, faces: Sequence[Dict[str, object]]
    ) -> List[List[Dict[str, object]]]:
        compatible: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
        for face in faces:
            embedding = np.asarray(face["embedding"])
            key = (str(face.get("model_name") or ""), int(embedding.size))
            compatible.setdefault(key, []).append(face)
        clusters = []
        for group in compatible.values():
            clusters.extend(self._discover_compatible_people(group))
        return clusters

    def _discover_compatible_people(
        self, faces: Sequence[Dict[str, object]]
    ) -> List[List[Dict[str, object]]]:
        if not faces:
            return []
        thresholds = self._thresholds_for(faces[0])
        eligible = [
            face
            for face in faces
            if float(face["quality"]) >= MINIMUM_IDENTITY_SEED_QUALITY
            and self._face_area(face) >= MINIMUM_IDENTITY_SEED_AREA
        ]
        strong_peer_counts: Dict[int, int] = {}
        for index, face in enumerate(eligible):
            for peer in eligible[index + 1 :]:
                if int(face["photo_id"]) == int(peer["photo_id"]):
                    continue
                if (
                    float(np.dot(face["embedding"], peer["embedding"]))
                    >= thresholds.new_person_strong_peer
                ):
                    face_id = int(face["id"])
                    peer_id = int(peer["id"])
                    strong_peer_counts[face_id] = strong_peer_counts.get(face_id, 0) + 1
                    strong_peer_counts[peer_id] = strong_peer_counts.get(peer_id, 0) + 1
        eligible = [
            face
            for face in eligible
            if strong_peer_counts.get(int(face["id"]), 0)
            >= NEW_PERSON_MIN_STRONG_PEERS
        ]
        clusters: List[List[Dict[str, object]]] = []
        for face in eligible:
            best_cluster = None
            best_similarity = -1.0
            for cluster in clusters:
                if any(
                    int(example["photo_id"]) == int(face["photo_id"])
                    for example in cluster
                ):
                    continue
                similarities = [
                    float(np.dot(face["embedding"], example["embedding"]))
                    for example in cluster
                ]
                if (
                    similarities
                    and min(similarities) >= thresholds.new_person_complete_link
                    and float(sum(similarities) / len(similarities)) > best_similarity
                ):
                    best_cluster = cluster
                    best_similarity = float(sum(similarities) / len(similarities))
            if best_cluster is None:
                clusters.append([face])
            else:
                best_cluster.append(face)
        supported_clusters = []
        for cluster in clusters:
            supported = [
                face
                for face in cluster
                if sum(
                    int(peer["photo_id"]) != int(face["photo_id"])
                    and float(np.dot(face["embedding"], peer["embedding"]))
                    >= thresholds.new_person_strong_peer
                    for peer in cluster
                    if int(peer["id"]) != int(face["id"])
                )
                >= NEW_PERSON_MIN_STRONG_PEERS
            ]
            if (
                len({int(face["photo_id"]) for face in supported})
                >= MINIMUM_NEW_PERSON_PHOTOS
            ):
                supported_clusters.append(supported)
        return self._merge_seed_clusters(supported_clusters)

    def _merge_seed_clusters(
        self, clusters: Sequence[Sequence[Dict[str, object]]]
    ) -> List[List[Dict[str, object]]]:
        """Join tight appearance groups using aggregate identity evidence."""

        merged = [list(cluster) for cluster in clusters]
        while True:
            best_pair = None
            best_score = -1.0
            for left_index, left in enumerate(merged):
                left_photos = {int(face["photo_id"]) for face in left}
                for right_index in range(left_index + 1, len(merged)):
                    right = merged[right_index]
                    if left_photos.intersection(
                        int(face["photo_id"]) for face in right
                    ):
                        continue
                    score = float(
                        np.dot(self._cluster_centroid(left), self._cluster_centroid(right))
                    )
                    thresholds = self._thresholds_for(left[0])
                    threshold = thresholds.seed_cluster_merge
                    larger, smaller = (
                        (left, right) if len(left) >= len(right) else (right, left)
                    )
                    if (
                        len({int(face["photo_id"]) for face in larger})
                        >= LARGE_CLUSTER_MIN_PHOTOS
                        and self._cluster_supported_by(
                            smaller,
                            larger,
                            thresholds.large_cluster_member_support,
                        )
                    ):
                        threshold = thresholds.large_cluster_merge
                    if score >= threshold and score > best_score:
                        best_pair = (left_index, right_index)
                        best_score = score
            if best_pair is None:
                return merged
            left_index, right_index = best_pair
            merged[left_index].extend(merged[right_index])
            del merged[right_index]

    @staticmethod
    def _cluster_supported_by(
        smaller: Sequence[Dict[str, object]],
        larger: Sequence[Dict[str, object]],
        minimum_similarity: float = LARGE_CLUSTER_MEMBER_SUPPORT,
    ) -> bool:
        supported = sum(
            max(
                float(np.dot(face["embedding"], example["embedding"]))
                for example in larger
            )
            >= minimum_similarity
            for face in smaller
        )
        return supported >= min(MINIMUM_NEW_PERSON_PHOTOS, len(smaller))

    @staticmethod
    def _cluster_centroid(cluster: Sequence[Dict[str, object]]) -> np.ndarray:
        centroid = np.mean(
            np.asarray([face["embedding"] for face in cluster], dtype=np.float32),
            axis=0,
        )
        norm = float(np.linalg.norm(centroid))
        return centroid / norm if norm > 1e-8 else centroid

    def _assign_recurring_faces(
        self,
        faces: Sequence[Dict[str, object]],
        prototypes: Dict[int, List[Dict[str, object]]],
        members: Dict[int, List[Dict[str, object]]],
        all_faces: Sequence[Dict[str, object]],
    ) -> int:
        """Fill gaps for a person who is demonstrably recurring in one folder."""

        face_photos_by_project: Dict[int, set[int]] = {}
        for face in all_faces:
            face_photos_by_project.setdefault(int(face["project_id"]), set()).add(
                int(face["photo_id"])
            )
        recurring = set()
        coverage_by_person_project: Dict[Tuple[int, int], float] = {}
        for person_id, examples in members.items():
            photos_by_project: Dict[int, set[int]] = {}
            for face in examples:
                photos_by_project.setdefault(int(face["project_id"]), set()).add(
                    int(face["photo_id"])
                )
            for project_id, photo_ids in photos_by_project.items():
                available = len(face_photos_by_project.get(project_id, ()))
                if (
                    len(photo_ids) >= RECURRING_PERSON_MIN_PHOTOS
                    and available
                    and len(photo_ids) / available >= RECURRING_PERSON_MIN_COVERAGE
                ):
                    recurring.add((person_id, project_id))
                    coverage_by_person_project[(person_id, project_id)] = (
                        len(photo_ids) / available
                    )
        if not recurring:
            return 0

        compatible_members: Dict[
            Tuple[int, str, int], List[Dict[str, object]]
        ] = {}
        for person_id, examples in members.items():
            for example in examples:
                embedding = np.asarray(example["embedding"])
                key = (
                    person_id,
                    str(example.get("model_name") or ""),
                    int(embedding.size),
                )
                compatible_members.setdefault(key, []).append(example)
        centroids = {
            key: self._cluster_centroid(examples)
            for key, examples in compatible_members.items()
            if examples
        }
        existing_photos = {
            person_id: {int(face["photo_id"]) for face in examples}
            for person_id, examples in members.items()
        }
        by_photo: Dict[int, List[Dict[str, object]]] = {}
        for face in faces:
            by_photo.setdefault(int(face["photo_id"]), []).append(face)

        assignments = []
        for photo_faces in by_photo.values():
            photo_assignments = []
            for face in photo_faces:
                thresholds = self._thresholds_for(face)
                face_key = (
                    str(face.get("model_name") or ""),
                    int(np.asarray(face["embedding"]).size),
                )
                best_by_person: Dict[int, float] = {}
                for (person_id, model_name, dimension), centroid in centroids.items():
                    if (model_name, dimension) != face_key or int(
                        face["photo_id"]
                    ) in existing_photos.get(person_id, set()):
                        continue
                    score = float(np.dot(face["embedding"], centroid))
                    best_by_person[person_id] = max(
                        score, best_by_person.get(person_id, -1.0)
                    )
                scores = sorted(
                    (score, person_id)
                    for person_id, score in best_by_person.items()
                )
                if not scores:
                    continue
                score, person_id = scores[-1]
                if (person_id, int(face["project_id"])) not in recurring:
                    continue
                if person_id in self.database.rejected_person_ids(int(face["id"])):
                    continue
                runner_up = scores[-2][0] if len(scores) > 1 else -1.0
                accepted = score >= thresholds.recurring_centroid and (
                    score >= thresholds.very_strong
                    or score - runner_up >= thresholds.recurring_margin
                )
                evidence_score = score
                coverage = coverage_by_person_project.get(
                    (person_id, int(face["project_id"])), 0.0
                )
                if not accepted and coverage >= DOMINANT_PERSON_MIN_COVERAGE:
                    compatible = [
                        example
                        for example in members.get(person_id, ())
                        if self._embedding_key(example) == face_key
                    ]
                    similarities = sorted(
                        (
                            float(np.dot(face["embedding"], example["embedding"]))
                            for example in compatible
                        ),
                        reverse=True,
                    )
                    if len(similarities) >= 3:
                        agreement = float(sum(similarities[:3]) / 3.0)
                        if agreement >= DOMINANT_PERSON_MEMBER_AGREEMENT:
                            accepted = True
                            evidence_score = agreement
                if accepted:
                    photo_assignments.append((evidence_score, face, person_id))

            # One dominant identity can occur only once in a photograph. When
            # several unknown faces remain, require a clear within-photo winner.
            photo_assignments.sort(reverse=True, key=lambda item: item[0])
            for candidate in photo_assignments:
                score, _face, person_id = candidate
                competitors = [
                    other_score
                    for other_score, _other_face, other_person_id in photo_assignments
                    if other_person_id == person_id and other_score != score
                ]
                if competitors and score - max(competitors) < DOMINANT_PERSON_FACE_MARGIN:
                    continue
                assignments.append(candidate)

        assigned_faces = set()
        assigned_person_photos = set()
        assigned = 0
        for score, face, person_id in sorted(assignments, reverse=True, key=lambda item: item[0]):
            face_id = int(face["id"])
            person_photo = (person_id, int(face["photo_id"]))
            if face_id in assigned_faces or person_photo in assigned_person_photos:
                continue
            self._assign(face, person_id, score, prototypes, members)
            assigned_faces.add(face_id)
            assigned_person_photos.add(person_photo)
            assigned += 1
        return assigned

    @staticmethod
    def _cluster_confidence(cluster: Sequence[Dict[str, object]]) -> float:
        similarities = [
            float(np.dot(left["embedding"], right["embedding"]))
            for index, left in enumerate(cluster)
            for right in cluster[index + 1 :]
        ]
        return max(0.0, min(1.0, min(similarities) if similarities else 1.0))

    @staticmethod
    def _face_area(face: Dict[str, object]) -> float:
        bounds = face["bounds"]
        return max(0.0, float(bounds[2]) * float(bounds[3]))

    def _best_person(
        self,
        face: Dict[str, object],
        prototypes: Dict[int, List[Dict[str, object]]],
    ) -> Tuple[Optional[int], float]:
        thresholds = self._thresholds_for(face)
        rejected = set(self.database.rejected_person_ids(int(face["id"])))
        candidates = []
        for person_id, examples in prototypes.items():
            if person_id in rejected or any(
                int(example["photo_id"]) == int(face["photo_id"])
                and int(example["id"]) != int(face["id"])
                for example in examples
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
                accepted = best >= thresholds.single_prototype + (
                    0.04 if named_identity else 0.0
                )
            else:
                accepted = (
                    best
                    >= thresholds.multi_prototype_best
                    + (0.04 if named_identity else 0.0)
                    and agreement
                    >= thresholds.multi_prototype_agreement
                    + (0.03 if named_identity else 0.0)
                )
            if accepted:
                candidates.append((agreement, best, person_id))
        if not candidates:
            return None, 0.0
        candidates.sort(reverse=True)
        agreement, best, person_id = candidates[0]
        runner_up = candidates[1][0] if len(candidates) > 1 else -1.0
        required_margin = thresholds.minimum_winner_margin + (
            0.02
            if any(
                bool(example.get("person_is_named"))
                for example in prototypes[person_id]
            )
            else 0.0
        )
        if (
            best < thresholds.very_strong
            and agreement - runner_up < required_margin
        ):
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
            person_id: cls._select_compatible_prototypes(examples)
            for person_id, examples in grouped.items()
        }

    @classmethod
    def _select_compatible_prototypes(
        cls, faces: Sequence[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        compatible: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
        for face in faces:
            embedding = np.asarray(face["embedding"])
            key = (str(face.get("model_name") or ""), int(embedding.size))
            compatible.setdefault(key, []).append(face)
        return [
            face
            for examples in compatible.values()
            for face in cls._select_prototypes(examples)
        ]

    @staticmethod
    def _member_map(
        faces: Iterable[Dict[str, object]],
    ) -> Dict[int, List[Dict[str, object]]]:
        grouped: Dict[int, List[Dict[str, object]]] = {}
        for face in faces:
            if face["person_id"] is not None:
                grouped.setdefault(int(face["person_id"]), []).append(face)
        return grouped

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
