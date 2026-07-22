from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from photochoose.assets import discover_logical_assets
from photochoose.models import (
    Decision,
    EyeEvidence,
    FaceEvidence,
    ObjectEvidence,
    Photo,
    PhotoView,
    Project,
)


class PhotoDatabase:
    """Small SQLite repository with one connection per operation.

    Connections are intentionally short-lived so scan workers and the UI thread
    never share a SQLite connection.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL UNIQUE,
                    is_open INTEGER NOT NULL DEFAULT 1 CHECK (is_open IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_opened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS photos (
                    id INTEGER PRIMARY KEY,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    relative_path TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    decision TEXT NOT NULL DEFAULT 'unreviewed'
                        CHECK (decision IN ('unreviewed', 'keep', 'review', 'reject')),
                    decision_source TEXT NOT NULL DEFAULT 'none',
                    is_best INTEGER NOT NULL DEFAULT 0 CHECK (is_best IN (0, 1)),
                    best_source TEXT NOT NULL DEFAULT 'none',
                    analysis_status TEXT NOT NULL DEFAULT 'pending',
                    focus_score REAL,
                    brightness REAL,
                    clipped_dark REAL,
                    clipped_bright REAL,
                    content_hash TEXT,
                    perceptual_hash TEXT,
                    auto_reasons TEXT NOT NULL DEFAULT '[]',
                    duplicate_of_id INTEGER REFERENCES photos(id) ON DELETE SET NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    eye_count INTEGER NOT NULL DEFAULT 0,
                    closed_eye_count INTEGER NOT NULL DEFAULT 0,
                    uncertain_eye_count INTEGER NOT NULL DEFAULT 0,
                    subject_model TEXT NOT NULL DEFAULT 'none',
                    face_confidence REAL,
                    eye_open_score REAL,
                    eye_focus_score REAL,
                    subject_regions TEXT NOT NULL DEFAULT '[]',
                    subject_focus_score REAL,
                    object_count INTEGER NOT NULL DEFAULT 0,
                    object_model TEXT NOT NULL DEFAULT 'none',
                    primary_subject_focus_score REAL,
                    subject_objects TEXT NOT NULL DEFAULT '[]',
                    similarity_group TEXT,
                    group_reason TEXT,
                    best_score REAL,
                    captured_at TEXT,
                    camera_make TEXT,
                    camera_model TEXT,
                    lens_model TEXT,
                    iso INTEGER,
                    exposure_seconds REAL,
                    aperture REAL,
                    focal_length REAL,
                    preview_relative_path TEXT,
                    cached_preview_path TEXT,
                    asset_paths TEXT NOT NULL DEFAULT '[]',
                    raw_count INTEGER NOT NULL DEFAULT 0,
                    sidecar_count INTEGER NOT NULL DEFAULT 0,
                    asset_key TEXT,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_id, relative_path)
                );

                CREATE INDEX IF NOT EXISTS photos_project_decision
                    ON photos(project_id, decision);
                CREATE INDEX IF NOT EXISTS photos_project_best
                    ON photos(project_id, is_best);
                """
            )
            self._add_missing_project_columns(connection)
            self._add_missing_photo_columns(connection)
            connection.commit()

    @staticmethod
    def _add_missing_project_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
        }
        if "is_open" not in existing:
            connection.execute("ALTER TABLE projects ADD COLUMN is_open INTEGER NOT NULL DEFAULT 1")

    @staticmethod
    def _add_missing_photo_columns(connection: sqlite3.Connection) -> None:
        """Apply additive migrations to databases created by earlier builds."""

        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(photos)").fetchall()
        }
        definitions = {
            "decision_source": "TEXT NOT NULL DEFAULT 'none'",
            "analysis_status": "TEXT NOT NULL DEFAULT 'pending'",
            "best_source": "TEXT NOT NULL DEFAULT 'none'",
            "focus_score": "REAL",
            "brightness": "REAL",
            "clipped_dark": "REAL",
            "clipped_bright": "REAL",
            "content_hash": "TEXT",
            "perceptual_hash": "TEXT",
            "auto_reasons": "TEXT NOT NULL DEFAULT '[]'",
            "duplicate_of_id": "INTEGER REFERENCES photos(id) ON DELETE SET NULL",
            "face_count": "INTEGER NOT NULL DEFAULT 0",
            "eye_count": "INTEGER NOT NULL DEFAULT 0",
            "closed_eye_count": "INTEGER NOT NULL DEFAULT 0",
            "uncertain_eye_count": "INTEGER NOT NULL DEFAULT 0",
            "subject_model": "TEXT NOT NULL DEFAULT 'none'",
            "face_confidence": "REAL",
            "eye_open_score": "REAL",
            "eye_focus_score": "REAL",
            "subject_regions": "TEXT NOT NULL DEFAULT '[]'",
            "subject_focus_score": "REAL",
            "object_count": "INTEGER NOT NULL DEFAULT 0",
            "object_model": "TEXT NOT NULL DEFAULT 'none'",
            "primary_subject_focus_score": "REAL",
            "subject_objects": "TEXT NOT NULL DEFAULT '[]'",
            "similarity_group": "TEXT",
            "group_reason": "TEXT",
            "best_score": "REAL",
            "captured_at": "TEXT",
            "camera_make": "TEXT",
            "camera_model": "TEXT",
            "lens_model": "TEXT",
            "iso": "INTEGER",
            "exposure_seconds": "REAL",
            "aperture": "REAL",
            "focal_length": "REAL",
            "preview_relative_path": "TEXT",
            "cached_preview_path": "TEXT",
            "asset_paths": "TEXT NOT NULL DEFAULT '[]'",
            "raw_count": "INTEGER NOT NULL DEFAULT 0",
            "sidecar_count": "INTEGER NOT NULL DEFAULT 0",
            "asset_key": "TEXT",
        }
        for name, definition in definitions.items():
            if name not in existing:
                connection.execute("ALTER TABLE photos ADD COLUMN {0} {1}".format(name, definition))

    def add_project(self, root_path: Path) -> Project:
        root = Path(root_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Project path is not a directory: {0}".format(root))

        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO projects(name, root_path)
                VALUES (?, ?)
                ON CONFLICT(root_path) DO UPDATE SET
                    name = excluded.name,
                    is_open = 1,
                    last_opened_at = CURRENT_TIMESTAMP
                """,
                (root.name or str(root), str(root)),
            )
            row = connection.execute(
                "SELECT id, name, root_path FROM projects WHERE root_path = ?",
                (str(root),),
            ).fetchone()
            connection.commit()
        return self._project_from_row(row)

    def list_projects(self) -> List[Project]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, name, root_path
                FROM projects
                WHERE is_open = 1
                ORDER BY last_opened_at DESC, name COLLATE NOCASE
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: int) -> Optional[Project]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, name, root_path FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return self._project_from_row(row) if row else None

    def close_project(self, project_id: int) -> Tuple[int, int]:
        """Hide a project and release only its regenerable RAW preview cache."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT cached_preview_path
                FROM photos
                WHERE project_id = ? AND cached_preview_path IS NOT NULL
                """,
                (project_id,),
            ).fetchall()
            connection.execute(
                "UPDATE photos SET cached_preview_path = NULL WHERE project_id = ?",
                (project_id,),
            )
            connection.execute("UPDATE projects SET is_open = 0 WHERE id = ?", (project_id,))
            connection.commit()

        cache_directory = (self.path.parent / "raw-previews").resolve()
        removed_count = 0
        removed_bytes = 0
        for row in rows:
            cache_path = Path(row["cached_preview_path"])
            try:
                resolved = cache_path.resolve()
                resolved.relative_to(cache_directory)
            except (OSError, ValueError):
                continue
            try:
                size = resolved.stat().st_size
                resolved.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
            removed_count += 1
            removed_bytes += size
        return removed_count, removed_bytes

    def scan_project(self, project_id: int) -> int:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Unknown project id: {0}".format(project_id))

        root = Path(project.root_path)
        if not root.is_dir():
            raise ValueError("Project folder is unavailable: {0}".format(root))

        with closing(self._connect()) as connection:
            existing_rows = connection.execute(
                "SELECT id, relative_path, asset_key FROM photos WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            for row in existing_rows:
                if not row["asset_key"]:
                    connection.execute(
                        "UPDATE photos SET asset_key = ? WHERE id = ?",
                        (self._legacy_asset_key(row["relative_path"]), row["id"]),
                    )
            connection.commit()
            existing_rows = connection.execute(
                "SELECT id, relative_path, asset_key FROM photos WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        existing_paths = [row["relative_path"] for row in existing_rows]
        existing_by_key = {row["asset_key"]: row for row in existing_rows}
        discovered = discover_logical_assets(
            root,
            self.path.parent / "raw-previews",
            existing_paths,
        )

        with closing(self._connect()) as connection:
            for metadata in discovered.values():
                existing_row = existing_by_key.get(metadata.asset_key)
                if (
                    metadata.relative_path not in existing_paths
                    and existing_row is not None
                    and existing_row["relative_path"] not in discovered
                ):
                    connection.execute(
                        "UPDATE photos SET relative_path = ? WHERE id = ?",
                        (metadata.relative_path, existing_row["id"]),
                    )
                    existing_paths.remove(existing_row["relative_path"])
                    existing_paths.append(metadata.relative_path)
            for relative_path, metadata in discovered.items():
                connection.execute(
                    """
                    INSERT INTO photos(
                        project_id, relative_path, extension, size_bytes, modified_ns,
                        captured_at, camera_make, camera_model, lens_model, iso,
                        exposure_seconds, aperture, focal_length, preview_relative_path,
                        cached_preview_path, asset_paths, raw_count, sidecar_count,
                        asset_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, relative_path) DO UPDATE SET
                        extension = excluded.extension,
                        size_bytes = excluded.size_bytes,
                        modified_ns = excluded.modified_ns,
                        preview_relative_path = excluded.preview_relative_path,
                        cached_preview_path = excluded.cached_preview_path,
                        asset_paths = excluded.asset_paths,
                        raw_count = excluded.raw_count,
                        sidecar_count = excluded.sidecar_count,
                        asset_key = excluded.asset_key,
                        captured_at = excluded.captured_at,
                        camera_make = excluded.camera_make,
                        camera_model = excluded.camera_model,
                        lens_model = excluded.lens_model,
                        iso = excluded.iso,
                        exposure_seconds = excluded.exposure_seconds,
                        aperture = excluded.aperture,
                        focal_length = excluded.focal_length,
                        analysis_status = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 'pending' ELSE photos.analysis_status END,
                        focus_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.focus_score END,
                        brightness = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.brightness END,
                        clipped_dark = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.clipped_dark END,
                        clipped_bright = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.clipped_bright END,
                        content_hash = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.content_hash END,
                        perceptual_hash = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.perceptual_hash END,
                        auto_reasons = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN '[]' ELSE photos.auto_reasons END,
                        duplicate_of_id = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.duplicate_of_id END,
                        face_count = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 0 ELSE photos.face_count END,
                        eye_count = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 0 ELSE photos.eye_count END,
                        closed_eye_count = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 0 ELSE photos.closed_eye_count END,
                        uncertain_eye_count = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 0 ELSE photos.uncertain_eye_count END,
                        subject_model = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 'none' ELSE photos.subject_model END,
                        face_confidence = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.face_confidence END,
                        eye_open_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.eye_open_score END,
                        eye_focus_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.eye_focus_score END,
                        subject_regions = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN '[]' ELSE photos.subject_regions END,
                        subject_focus_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.subject_focus_score END,
                        object_count = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 0 ELSE photos.object_count END,
                        object_model = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN 'none' ELSE photos.object_model END,
                        primary_subject_focus_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.primary_subject_focus_score END,
                        subject_objects = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN '[]' ELSE photos.subject_objects END,
                        similarity_group = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.similarity_group END,
                        group_reason = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.group_reason END,
                        best_score = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.best_score END,
                        is_best = CASE
                            WHEN (photos.size_bytes != excluded.size_bytes
                               OR photos.modified_ns != excluded.modified_ns)
                              AND photos.best_source = 'automatic'
                            THEN 0 ELSE photos.is_best END,
                        best_source = CASE
                            WHEN (photos.size_bytes != excluded.size_bytes
                               OR photos.modified_ns != excluded.modified_ns)
                              AND photos.best_source = 'automatic'
                            THEN 'none' ELSE photos.best_source END,
                        decision = CASE
                            WHEN (photos.size_bytes != excluded.size_bytes
                               OR photos.modified_ns != excluded.modified_ns)
                              AND photos.decision_source = 'automatic'
                            THEN 'unreviewed' ELSE photos.decision END,
                        decision_source = CASE
                            WHEN (photos.size_bytes != excluded.size_bytes
                               OR photos.modified_ns != excluded.modified_ns)
                              AND photos.decision_source = 'automatic'
                            THEN 'none' ELSE photos.decision_source END,
                        indexed_at = CURRENT_TIMESTAMP
                    """,
                    (
                        project_id,
                        relative_path,
                        metadata.extension,
                        metadata.size_bytes,
                        metadata.modified_ns,
                        metadata.metadata.captured_at,
                        metadata.metadata.camera_make,
                        metadata.metadata.camera_model,
                        metadata.metadata.lens_model,
                        metadata.metadata.iso,
                        metadata.metadata.exposure_seconds,
                        metadata.metadata.aperture,
                        metadata.metadata.focal_length,
                        metadata.preview_relative_path,
                        metadata.cached_preview_path,
                        json.dumps(metadata.asset_paths),
                        metadata.raw_count,
                        metadata.sidecar_count,
                        metadata.asset_key,
                    ),
                )

            existing = connection.execute(
                "SELECT relative_path FROM photos WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            missing = [
                row["relative_path"] for row in existing if row["relative_path"] not in discovered
            ]
            if missing:
                connection.executemany(
                    "DELETE FROM photos WHERE project_id = ? AND relative_path = ?",
                    [(project_id, relative_path) for relative_path in missing],
                )

            connection.execute(
                "UPDATE projects SET last_opened_at = CURRENT_TIMESTAMP WHERE id = ?",
                (project_id,),
            )
            connection.commit()
        self._cleanup_preview_cache()
        return len(discovered)

    def _cleanup_preview_cache(self) -> None:
        cache_directory = self.path.parent / "raw-previews"
        if not cache_directory.is_dir():
            return
        with closing(self._connect()) as connection:
            referenced = {
                row["cached_preview_path"]
                for row in connection.execute(
                    """
                    SELECT cached_preview_path FROM photos
                    WHERE cached_preview_path IS NOT NULL
                    """
                ).fetchall()
            }
        for cache_path in cache_directory.glob("*.jpg"):
            if str(cache_path) not in referenced:
                try:
                    cache_path.unlink()
                except OSError:
                    continue

    def list_photos(self, project_id: int, view: PhotoView) -> List[Photo]:
        clauses = ["p.project_id = ?"]
        parameters: List[object] = [project_id]
        if view == PhotoView.BEST:
            clauses.append("p.is_best = 1")
        elif view != PhotoView.ALL:
            clauses.append("p.decision = ?")
            parameters.append(view.value)

        where = " AND ".join(clauses)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT p.id, p.project_id, p.relative_path, p.decision, p.is_best,
                       p.decision_source, p.analysis_status, p.focus_score,
                       p.brightness, p.clipped_dark, p.clipped_bright,
                       p.auto_reasons, p.duplicate_of_id, p.best_source,
                       p.perceptual_hash, p.face_count, p.eye_count,
                       p.closed_eye_count, p.uncertain_eye_count, p.subject_model,
                       p.face_confidence, p.eye_open_score, p.eye_focus_score,
                       p.subject_regions,
                       p.subject_focus_score, p.object_count, p.object_model,
                       p.primary_subject_focus_score, p.subject_objects,
                       p.similarity_group, p.group_reason,
                       p.best_score, p.captured_at, p.camera_make, p.camera_model,
                       p.lens_model, p.iso, p.exposure_seconds, p.aperture,
                       p.focal_length, p.preview_relative_path,
                       p.cached_preview_path, p.asset_paths, p.raw_count,
                       p.sidecar_count,
                       projects.root_path
                FROM photos AS p
                JOIN projects ON projects.id = p.project_id
                WHERE {where}
                ORDER BY p.relative_path COLLATE NOCASE
                """.format(where=where),
                parameters,
            ).fetchall()
        return [self._photo_from_row(row) for row in rows]

    def set_decision(self, photo_ids: Sequence[int], decision: Decision) -> None:
        if not photo_ids:
            return
        placeholders = ",".join("?" for _ in photo_ids)
        with closing(self._connect()) as connection:
            source = "none" if decision == Decision.UNREVIEWED else "manual"
            connection.execute(
                """
                UPDATE photos SET decision = ?, decision_source = ?
                WHERE id IN ({0})
                """.format(placeholders),
                [decision.value, source] + list(photo_ids),
            )
            connection.commit()

    def apply_analysis(
        self,
        photo_id: int,
        decision: Decision,
        focus_score: Optional[float],
        brightness: Optional[float],
        clipped_dark: Optional[float],
        clipped_bright: Optional[float],
        content_hash: Optional[str],
        reasons: Sequence[str],
        duplicate_of_id: Optional[int],
        status: str = "complete",
    ) -> None:
        """Persist analysis without overwriting an explicit manual decision."""

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET
                    analysis_status = ?,
                    focus_score = ?,
                    brightness = ?,
                    clipped_dark = ?,
                    clipped_bright = ?,
                    content_hash = ?,
                    auto_reasons = ?,
                    duplicate_of_id = ?,
                    decision = CASE
                        WHEN decision_source = 'manual' THEN decision
                        ELSE ?
                    END,
                    decision_source = CASE
                        WHEN decision_source = 'manual' THEN decision_source
                        ELSE 'automatic'
                    END
                WHERE id = ?
                """,
                (
                    status,
                    focus_score,
                    brightness,
                    clipped_dark,
                    clipped_bright,
                    content_hash,
                    json.dumps(list(reasons)),
                    duplicate_of_id,
                    decision.value,
                    photo_id,
                ),
            )
            connection.commit()

    def toggle_best(self, photo_ids: Sequence[int]) -> None:
        if not photo_ids:
            return
        placeholders = ",".join("?" for _ in photo_ids)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos
                SET is_best = CASE is_best WHEN 1 THEN 0 ELSE 1 END,
                    best_source = 'manual'
                WHERE id IN ({0})
                """.format(placeholders),
                list(photo_ids),
            )
            connection.commit()

    def apply_subject_analysis(
        self,
        photo_id: int,
        perceptual_hash: Optional[str],
        face_count: int,
        eye_count: int,
        closed_eye_count: int,
        uncertain_eye_count: int,
        subject_focus_score: Optional[float],
        subject_model: str,
        face_confidence: Optional[float],
        eye_open_score: Optional[float],
        eye_focus_score: Optional[float] = None,
        subject_regions: Sequence[FaceEvidence] = (),
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET
                    perceptual_hash = ?, face_count = ?, eye_count = ?,
                    closed_eye_count = ?, uncertain_eye_count = ?,
                    subject_focus_score = ?, subject_model = ?,
                    face_confidence = ?, eye_open_score = ?,
                    eye_focus_score = ?, subject_regions = ?
                WHERE id = ?
                """,
                (
                    perceptual_hash,
                    face_count,
                    eye_count,
                    closed_eye_count,
                    uncertain_eye_count,
                    subject_focus_score,
                    subject_model,
                    face_confidence,
                    eye_open_score,
                    eye_focus_score,
                    self._encode_subject_regions(subject_regions),
                    photo_id,
                ),
            )
            connection.commit()

    def apply_object_analysis(
        self,
        photo_id: int,
        object_count: int,
        object_model: str,
        primary_subject_focus_score: Optional[float],
        subject_objects: Sequence[ObjectEvidence],
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET
                    object_count = ?, object_model = ?,
                    primary_subject_focus_score = ?, subject_objects = ?
                WHERE id = ?
                """,
                (
                    object_count,
                    object_model,
                    primary_subject_focus_score,
                    self._encode_subject_objects(subject_objects),
                    photo_id,
                ),
            )
            connection.commit()

    def reset_automatic_groups(self, project_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET
                    similarity_group = NULL,
                    group_reason = NULL,
                    best_score = NULL,
                    is_best = CASE
                        WHEN best_source = 'automatic' THEN 0 ELSE is_best END,
                    best_source = CASE
                        WHEN best_source = 'automatic' THEN 'none' ELSE best_source END
                WHERE project_id = ?
                """,
                (project_id,),
            )
            connection.commit()

    def set_similarity_group(
        self, photo_id: int, group_id: str, score: float, reason: str
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos
                SET similarity_group = ?, best_score = ?, group_reason = ?
                WHERE id = ?
                """,
                (group_id, score, reason, photo_id),
            )
            connection.commit()

    def set_automatic_best(self, photo_id: int) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET is_best = 1, best_source = 'automatic'
                WHERE id = ? AND best_source != 'manual'
                """,
                (photo_id,),
            )
            connection.commit()

    def reject_weaker_group_photo(
        self,
        photo_id: int,
        group_id: str,
        winner_filename: str,
        score_gap: float,
    ) -> None:
        """Reject an automatic Review only when a much stronger grouped frame exists."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT decision, decision_source, auto_reasons FROM photos WHERE id = ?",
                (photo_id,),
            ).fetchone()
            if (
                row is None
                or row["decision"] != Decision.REVIEW.value
                or row["decision_source"] == "manual"
            ):
                return
            reasons = list(json.loads(row["auto_reasons"] or "[]"))
            reasons.append(
                "Weaker technical frame in {0}; {1} ranked {2:.1f} points higher".format(
                    group_id, winner_filename, score_gap
                )
            )
            connection.execute(
                """
                UPDATE photos
                SET decision = 'reject', decision_source = 'automatic', auto_reasons = ?
                WHERE id = ? AND decision = 'review' AND decision_source != 'manual'
                """,
                (json.dumps(reasons), photo_id),
            )
            connection.commit()

    def counts(self, project_id: int) -> Dict[str, int]:
        counts = {view.value: 0 for view in PhotoView}
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT decision, COUNT(*) AS total
                FROM photos
                WHERE project_id = ?
                GROUP BY decision
                """,
                (project_id,),
            ).fetchall()
            best = connection.execute(
                "SELECT COUNT(*) AS total FROM photos WHERE project_id = ? AND is_best = 1",
                (project_id,),
            ).fetchone()
        for row in rows:
            counts[row["decision"]] = row["total"]
            counts[PhotoView.ALL.value] += row["total"]
        counts[PhotoView.BEST.value] = best["total"]
        return counts

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> Project:
        return Project(id=row["id"], name=row["name"], root_path=row["root_path"])

    @staticmethod
    def _legacy_asset_key(relative_path: str) -> str:
        path = Path(relative_path)
        parent = path.parent.as_posix().casefold()
        stem = path.stem.casefold()
        return "{0}/{1}".format(parent, stem) if parent != "." else stem

    @staticmethod
    def _photo_from_row(row: sqlite3.Row) -> Photo:
        root_path = Path(row["root_path"])
        absolute_path = root_path / Path(row["relative_path"])
        preview_path = None
        if row["cached_preview_path"]:
            preview_path = row["cached_preview_path"]
        elif row["preview_relative_path"]:
            preview_path = str(root_path / Path(row["preview_relative_path"]))
        return Photo(
            id=row["id"],
            project_id=row["project_id"],
            relative_path=row["relative_path"],
            absolute_path=str(absolute_path),
            decision=Decision(row["decision"]),
            is_best=bool(row["is_best"]),
            decision_source=row["decision_source"],
            analysis_status=row["analysis_status"],
            focus_score=row["focus_score"],
            brightness=row["brightness"],
            clipped_dark=row["clipped_dark"],
            clipped_bright=row["clipped_bright"],
            auto_reasons=tuple(json.loads(row["auto_reasons"] or "[]")),
            duplicate_of_id=row["duplicate_of_id"],
            best_source=row["best_source"],
            perceptual_hash=row["perceptual_hash"],
            face_count=row["face_count"],
            eye_count=row["eye_count"],
            closed_eye_count=row["closed_eye_count"],
            uncertain_eye_count=row["uncertain_eye_count"],
            subject_model=row["subject_model"],
            face_confidence=row["face_confidence"],
            eye_open_score=row["eye_open_score"],
            eye_focus_score=row["eye_focus_score"],
            subject_regions=PhotoDatabase._decode_subject_regions(
                row["subject_regions"]
            ),
            subject_focus_score=row["subject_focus_score"],
            object_count=row["object_count"],
            object_model=row["object_model"],
            primary_subject_focus_score=row["primary_subject_focus_score"],
            subject_objects=PhotoDatabase._decode_subject_objects(
                row["subject_objects"]
            ),
            similarity_group=row["similarity_group"],
            best_score=row["best_score"],
            group_reason=row["group_reason"],
            captured_at=row["captured_at"],
            camera_make=row["camera_make"],
            camera_model=row["camera_model"],
            lens_model=row["lens_model"],
            iso=row["iso"],
            exposure_seconds=row["exposure_seconds"],
            aperture=row["aperture"],
            focal_length=row["focal_length"],
            preview_absolute_path=preview_path,
            asset_paths=tuple(json.loads(row["asset_paths"] or "[]")),
            raw_count=row["raw_count"],
            sidecar_count=row["sidecar_count"],
        )

    @staticmethod
    def _encode_subject_regions(regions: Sequence[FaceEvidence]) -> str:
        return json.dumps(
            [
                {
                    "bounds": list(region.bounds),
                    "confidence": region.confidence,
                    "focus_score": region.focus_score,
                    "important": region.important,
                    "eyes": [
                        {
                            "state": eye.state,
                            "aperture": eye.aperture,
                            "openness_score": eye.openness_score,
                            "focus_score": eye.focus_score,
                            "landmarks": [list(point) for point in eye.landmarks],
                        }
                        for eye in region.eyes
                    ],
                }
                for region in regions
            ],
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_subject_regions(value: str) -> Tuple[FaceEvidence, ...]:
        try:
            decoded = json.loads(value or "[]")
            return tuple(
                FaceEvidence(
                    bounds=tuple(float(number) for number in region["bounds"]),
                    confidence=float(region["confidence"]),
                    focus_score=(
                        float(region["focus_score"])
                        if region.get("focus_score") is not None
                        else None
                    ),
                    important=bool(region.get("important", True)),
                    eyes=tuple(
                        EyeEvidence(
                            state=str(eye["state"]),
                            aperture=(
                                float(eye["aperture"])
                                if eye.get("aperture") is not None
                                else None
                            ),
                            openness_score=(
                                float(eye["openness_score"])
                                if eye.get("openness_score") is not None
                                else None
                            ),
                            focus_score=(
                                float(eye["focus_score"])
                                if eye.get("focus_score") is not None
                                else None
                            ),
                            landmarks=tuple(
                                (float(point[0]), float(point[1]))
                                for point in eye.get("landmarks", [])
                            ),
                        )
                        for eye in region.get("eyes", [])
                    ),
                )
                for region in decoded
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ()

    @staticmethod
    def _encode_subject_objects(objects: Sequence[ObjectEvidence]) -> str:
        return json.dumps(
            [
                {
                    "label": item.label,
                    "bounds": list(item.bounds),
                    "confidence": item.confidence,
                    "focus_score": item.focus_score,
                    "importance": item.importance,
                }
                for item in objects
            ],
            separators=(",", ":"),
        )

    @staticmethod
    def _decode_subject_objects(value: str) -> Tuple[ObjectEvidence, ...]:
        try:
            decoded = json.loads(value or "[]")
            return tuple(
                ObjectEvidence(
                    label=str(item["label"]),
                    bounds=tuple(float(number) for number in item["bounds"]),
                    confidence=float(item["confidence"]),
                    focus_score=(
                        float(item["focus_score"])
                        if item.get("focus_score") is not None
                        else None
                    ),
                    importance=str(item["importance"]),
                )
                for item in decoded
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return ()
