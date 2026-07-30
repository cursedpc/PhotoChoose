"""Measure the packaged identity encoder against an existing local catalog.

This tool is read-only: it decodes the recorded face regions and reports score
distributions without changing the catalog or its assignments.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from photochoose.database import PhotoDatabase
from photochoose.identity_recognition import (
    OPENVINO_REID_MODEL_PATH,
    IdentityMatcher,
    OpenVINOIdentityEncoder,
    SFaceIdentityEncoder,
)
from photochoose.image_loader import ImageDecodeError, decode_analysis_bgr


@dataclass(frozen=True)
class EncodedFace:
    face_id: int
    photo_id: int
    person_id: int | None
    embedding: np.ndarray
    model_name: str


def _percentiles(label: str, values: list[float]) -> None:
    if not values:
        print(f"{label}: no samples")
        return
    samples = np.asarray(values, dtype=np.float32)
    points = np.percentile(samples, (0, 1, 5, 25, 50, 75, 95, 99, 100))
    rendered = " ".join(
        f"p{percentile:02d}={value:.4f}"
        for percentile, value in zip((0, 1, 5, 25, 50, 75, 95, 99, 100), points)
    )
    print(f"{label} ({len(values)}): {rendered}")


def _rows(database_path: Path) -> list[sqlite3.Row]:
    uri = f"file:{database_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(
            """
            SELECT f.id, f.photo_id, f.face_index, f.bounds, f.landmarks,
                   f.person_id, p.relative_path, projects.root_path
            FROM face_instances AS f
            JOIN photos AS p ON p.id = f.photo_id
            JOIN projects ON projects.id = p.project_id
            ORDER BY f.id
            """
        ).fetchall()
    finally:
        connection.close()


def _detected_face(row: sqlite3.Row, width: int, height: int) -> tuple[np.ndarray, tuple]:
    import json

    bounds = tuple(float(value) for value in json.loads(row["bounds"]))
    landmarks = tuple(
        (float(point[0]), float(point[1]))
        for point in json.loads(row["landmarks"])
    )
    face = np.zeros(15, dtype=np.float32)
    face[:4] = (
        bounds[0] * width,
        bounds[1] * height,
        bounds[2] * width,
        bounds[3] * height,
    )
    for index, point in enumerate(landmarks[:5]):
        face[4 + index * 2] = point[0] * width
        face[5 + index * 2] = point[1] * height
    face[14] = 1.0
    return face, bounds


def encode_catalog(database_path: Path, encoder) -> list[EncodedFace]:
    encoded: list[EncodedFace] = []
    skipped = 0
    started = time.monotonic()
    for row in _rows(database_path):
        path = Path(row["root_path"]) / row["relative_path"]
        if not path.is_file():
            skipped += 1
            continue
        try:
            image = decode_analysis_bgr(path)
        except (ImageDecodeError, OSError):
            skipped += 1
            continue
        face, bounds = _detected_face(row, image.shape[1], image.shape[0])
        evidence = encoder.encode(
            image,
            face,
            int(row["face_index"]),
            bounds,
            focus_score=None,
        )
        if evidence is None:
            skipped += 1
            continue
        encoded.append(
            EncodedFace(
                face_id=int(row["id"]),
                photo_id=int(row["photo_id"]),
                person_id=(
                    int(row["person_id"]) if row["person_id"] is not None else None
                ),
                embedding=np.asarray(evidence.embedding, dtype=np.float32),
                model_name=evidence.model_name,
            )
        )
    elapsed = time.monotonic() - started
    print(
        f"Encoded {len(encoded)} faces; skipped {skipped}; "
        f"{elapsed:.1f}s total ({elapsed / max(1, len(encoded)):.3f}s/face)"
    )
    return encoded


def report(faces: list[EncodedFace]) -> None:
    within_person: list[float] = []
    same_photo_different_person: list[float] = []
    different_faces_in_one_photo: list[float] = []
    for index, left in enumerate(faces):
        for right in faces[index + 1 :]:
            score = float(np.dot(left.embedding, right.embedding))
            if left.person_id is not None and left.person_id == right.person_id:
                within_person.append(score)
            if left.photo_id == right.photo_id:
                different_faces_in_one_photo.append(score)
            if (
                left.photo_id == right.photo_id
                and left.person_id is not None
                and right.person_id is not None
                and left.person_id != right.person_id
            ):
                same_photo_different_person.append(score)
    _percentiles("within current person", within_person)
    _percentiles("different detections in one photo", different_faces_in_one_photo)
    _percentiles("different current people in one photo", same_photo_different_person)

    grouped: dict[int, list[EncodedFace]] = {}
    for face in faces:
        if face.person_id is not None:
            grouped.setdefault(face.person_id, []).append(face)
    for person_id, members in sorted(grouped.items(), key=lambda item: -len(item[1]))[:12]:
        centroid = np.mean([member.embedding for member in members], axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
        scores = [float(np.dot(member.embedding, centroid)) for member in members]
        _percentiles(f"person {person_id} to centroid", scores)


def regroup_copy(database_path: Path, faces: list[EncodedFace]) -> None:
    """Replace embeddings and regroup the explicitly supplied disposable copy."""

    connection = sqlite3.connect(database_path)
    try:
        face_ids = {face.face_id for face in faces}
        for face in faces:
            embedding = np.asarray(face.embedding, dtype=np.float32)
            connection.execute(
                """
                UPDATE face_instances SET embedding = ?, embedding_dimension = ?,
                    model_name = ?
                WHERE id = ?
                """,
                (
                    sqlite3.Binary(embedding.tobytes()),
                    int(embedding.size),
                    face.model_name,
                    face.face_id,
                ),
            )
        if face_ids:
            placeholders = ",".join("?" for _face_id in face_ids)
            connection.execute(
                f"DELETE FROM face_instances WHERE id NOT IN ({placeholders})",
                tuple(sorted(face_ids)),
            )
        connection.commit()
    finally:
        connection.close()

    database = PhotoDatabase(database_path)
    stats = IdentityMatcher(database).assign_unknown_faces()
    people = database.list_people()
    print(
        "Regrouped copy: assigned={0} people={1} unknown={2}".format(
            stats.assigned, len(people), stats.unknown
        )
    )
    print(
        "Largest groups:",
        [(person.display_name, person.photo_count) for person in people[:20]],
    )
    report_remaining(database)


def report_remaining(database: PhotoDatabase) -> None:
    people = database.list_people()
    if people:
        all_faces = database.list_identity_faces()
        largest = max(people, key=lambda person: person.face_count)
        members = [
            face for face in all_faces if face["person_id"] == largest.id
        ]
        member_photos = {int(face["photo_id"]) for face in members}
        centroid = IdentityMatcher._cluster_centroid(members)
        candidates = [
            float(np.dot(face["embedding"], centroid))
            for face in all_faces
            if face["person_id"] is None
            and int(face["photo_id"]) not in member_photos
            and np.asarray(face["embedding"]).shape == centroid.shape
        ]
        _percentiles(
            f"remaining candidates for {largest.display_name}", candidates
        )
        print(
            "Remaining candidate counts:",
            {
                threshold: sum(score >= threshold for score in candidates)
                for threshold in (0.25, 0.30, 0.35, 0.38, 0.40, 0.45, 0.50)
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--encoder", choices=("openvino", "sface"), default="openvino")
    parser.add_argument("--openvino-model", type=Path)
    parser.add_argument(
        "--regroup-copy",
        action="store_true",
        help="write embeddings and grouping only to the supplied disposable database copy",
    )
    arguments = parser.parse_args()
    encoder = (
        SFaceIdentityEncoder()
        if arguments.encoder == "sface"
        else OpenVINOIdentityEncoder(
            arguments.openvino_model
            if arguments.openvino_model
            else OPENVINO_REID_MODEL_PATH
        )
    )
    faces = encode_catalog(arguments.database, encoder)
    report(faces)
    if arguments.regroup_copy:
        regroup_copy(arguments.database, faces)


if __name__ == "__main__":
    main()
