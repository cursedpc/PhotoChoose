from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from photochoose.analysis_profiles import ANALYSIS_VERSION, get_analysis_profile
from photochoose.assets import discover_logical_assets
from photochoose.models import (
    Decision,
    EyeEvidence,
    FaceEvidence,
    IdentityFaceEvidence,
    ObjectEvidence,
    PersonFolder,
    PersonIdentity,
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
                    analysis_profile TEXT NOT NULL DEFAULT 'safe'
                        CHECK (analysis_profile IN ('safe', 'balanced', 'strict')),
                    best_count_per_group INTEGER NOT NULL DEFAULT 1
                        CHECK (best_count_per_group BETWEEN 1 AND 5),
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
                    analysis_profile TEXT,
                    analysis_version TEXT,
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
                    motion_count INTEGER NOT NULL DEFAULT 0,
                    burst_id TEXT,
                    live_photo_id TEXT,
                    asset_key TEXT,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(project_id, relative_path)
                );

                CREATE INDEX IF NOT EXISTS photos_project_decision
                    ON photos(project_id, decision);
                CREATE INDEX IF NOT EXISTS photos_project_best
                    ON photos(project_id, is_best);

                CREATE TABLE IF NOT EXISTS manual_overrides (
                    id INTEGER PRIMARY KEY,
                    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    action_type TEXT NOT NULL CHECK (action_type IN ('decision', 'best')),
                    previous_value TEXT NOT NULL,
                    previous_source TEXT NOT NULL,
                    new_value TEXT NOT NULL,
                    analysis_profile TEXT,
                    analysis_version TEXT,
                    photo_path TEXT,
                    auto_reasons TEXT NOT NULL DEFAULT '[]',
                    focus_score REAL,
                    brightness REAL,
                    clipped_dark REAL,
                    clipped_bright REAL,
                    face_count INTEGER,
                    eye_count INTEGER,
                    closed_eye_count INTEGER,
                    uncertain_eye_count INTEGER,
                    subject_focus_score REAL,
                    face_confidence REAL,
                    eye_open_score REAL,
                    eye_focus_score REAL,
                    object_count INTEGER,
                    primary_subject_focus_score REAL,
                    best_score REAL,
                    similarity_group TEXT,
                    best_count_per_group INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS manual_overrides_project_created
                    ON manual_overrides(project_id, created_at);

                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY,
                    display_name TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS face_instances (
                    id INTEGER PRIMARY KEY,
                    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                    face_index INTEGER NOT NULL,
                    bounds TEXT NOT NULL,
                    landmarks TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    quality REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
                    assignment_source TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (assignment_source IN ('unknown', 'automatic', 'manual')),
                    assignment_confidence REAL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(photo_id, face_index)
                );

                CREATE INDEX IF NOT EXISTS face_instances_person
                    ON face_instances(person_id);
                CREATE INDEX IF NOT EXISTS face_instances_photo
                    ON face_instances(photo_id);

                CREATE TABLE IF NOT EXISTS identity_rejections (
                    face_id INTEGER NOT NULL REFERENCES face_instances(id) ON DELETE CASCADE,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(face_id, person_id)
                );

                CREATE TABLE IF NOT EXISTS identity_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._add_missing_project_columns(connection)
            self._add_missing_photo_columns(connection)
            self._add_missing_override_columns(connection)
            self._mark_outdated_analysis(connection)
            self._remove_automatic_best(connection)
            connection.commit()

    @staticmethod
    def _remove_automatic_best(connection: sqlite3.Connection) -> None:
        """Migrate legacy automatic Best flags to the new manual-only behavior."""

        connection.execute(
            """
            UPDATE photos SET is_best = 0, best_source = 'none'
            WHERE best_source = 'automatic'
            """
        )

    @staticmethod
    def _add_missing_project_columns(connection: sqlite3.Connection) -> None:
        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
        }
        if "is_open" not in existing:
            connection.execute("ALTER TABLE projects ADD COLUMN is_open INTEGER NOT NULL DEFAULT 1")
        if "analysis_profile" not in existing:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN analysis_profile "
                "TEXT NOT NULL DEFAULT 'safe'"
            )
        if "best_count_per_group" not in existing:
            connection.execute(
                "ALTER TABLE projects ADD COLUMN best_count_per_group "
                "INTEGER NOT NULL DEFAULT 1"
            )

    @staticmethod
    def _add_missing_photo_columns(connection: sqlite3.Connection) -> None:
        """Apply additive migrations to databases created by earlier builds."""

        existing = {
            row["name"] for row in connection.execute("PRAGMA table_info(photos)").fetchall()
        }
        definitions = {
            "decision_source": "TEXT NOT NULL DEFAULT 'none'",
            "analysis_status": "TEXT NOT NULL DEFAULT 'pending'",
            "analysis_profile": "TEXT",
            "analysis_version": "TEXT",
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
            "motion_count": "INTEGER NOT NULL DEFAULT 0",
            "burst_id": "TEXT",
            "live_photo_id": "TEXT",
            "asset_key": "TEXT",
        }
        for name, definition in definitions.items():
            if name not in existing:
                connection.execute("ALTER TABLE photos ADD COLUMN {0} {1}".format(name, definition))

    @staticmethod
    def _add_missing_override_columns(connection: sqlite3.Connection) -> None:
        """Extend calibration snapshots created by earlier profile builds."""

        existing = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(manual_overrides)"
            ).fetchall()
        }
        definitions = {
            "photo_path": "TEXT",
            "auto_reasons": "TEXT NOT NULL DEFAULT '[]'",
            "focus_score": "REAL",
            "brightness": "REAL",
            "clipped_dark": "REAL",
            "clipped_bright": "REAL",
            "face_count": "INTEGER",
            "eye_count": "INTEGER",
            "closed_eye_count": "INTEGER",
            "uncertain_eye_count": "INTEGER",
            "subject_focus_score": "REAL",
            "face_confidence": "REAL",
            "eye_open_score": "REAL",
            "eye_focus_score": "REAL",
            "object_count": "INTEGER",
            "primary_subject_focus_score": "REAL",
            "best_score": "REAL",
            "similarity_group": "TEXT",
            "best_count_per_group": "INTEGER",
        }
        for name, definition in definitions.items():
            if name not in existing:
                connection.execute(
                    "ALTER TABLE manual_overrides ADD COLUMN {0} {1}".format(
                        name, definition
                    )
                )

    @staticmethod
    def _mark_outdated_analysis(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            UPDATE photos SET analysis_status = 'stale'
            WHERE analysis_status != 'pending'
              AND COALESCE(analysis_version, '') != ?
            """,
            (ANALYSIS_VERSION,),
        )

    def add_project(self, root_path: Path) -> Project:
        root = Path(root_path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("Folder path is not a directory: {0}".format(root))

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
                "SELECT id, name, root_path, analysis_profile, best_count_per_group "
                "FROM projects WHERE root_path = ?",
                (str(root),),
            ).fetchone()
            connection.commit()
        return self._project_from_row(row)

    def list_projects(self) -> List[Project]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, name, root_path, analysis_profile, best_count_per_group
                FROM projects
                WHERE is_open = 1
                ORDER BY last_opened_at DESC, name COLLATE NOCASE
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def list_all_projects(self) -> List[Project]:
        """Return open and closed folders retained in the local catalog."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, name, root_path, analysis_profile, best_count_per_group
                FROM projects
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def get_project(self, project_id: int) -> Optional[Project]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT id, name, root_path, analysis_profile, best_count_per_group "
                "FROM projects WHERE id = ?",
                (project_id,),
            ).fetchone()
        return self._project_from_row(row) if row else None

    def close_project(self, project_id: int) -> Tuple[int, int]:
        """Hide a folder and release only its regenerable RAW preview cache."""

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

    def scan_project(
        self,
        project_id: int,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError("Unknown folder id: {0}".format(project_id))

        root = Path(project.root_path)
        if not root.is_dir():
            raise ValueError("Photo folder is unavailable: {0}".format(root))

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
            progress,
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
                        motion_count, burst_id, live_photo_id, asset_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, relative_path) DO UPDATE SET
                        extension = excluded.extension,
                        size_bytes = excluded.size_bytes,
                        modified_ns = excluded.modified_ns,
                        preview_relative_path = excluded.preview_relative_path,
                        cached_preview_path = excluded.cached_preview_path,
                        asset_paths = excluded.asset_paths,
                        raw_count = excluded.raw_count,
                        sidecar_count = excluded.sidecar_count,
                        motion_count = excluded.motion_count,
                        burst_id = excluded.burst_id,
                        live_photo_id = excluded.live_photo_id,
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
                        analysis_profile = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.analysis_profile END,
                        analysis_version = CASE
                            WHEN photos.size_bytes != excluded.size_bytes
                              OR photos.modified_ns != excluded.modified_ns
                            THEN NULL ELSE photos.analysis_version END,
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
                        metadata.motion_count,
                        metadata.metadata.burst_id,
                        metadata.metadata.live_photo_id,
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
                SELECT p.*, projects.root_path
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
            rows = connection.execute(
                """
                SELECT id, project_id, relative_path, decision, decision_source,
                       analysis_profile, analysis_version, auto_reasons,
                       focus_score, brightness, clipped_dark, clipped_bright,
                       face_count, eye_count, closed_eye_count, uncertain_eye_count,
                       subject_focus_score, face_confidence, eye_open_score,
                       eye_focus_score, object_count, primary_subject_focus_score,
                       best_score, similarity_group,
                       (SELECT best_count_per_group FROM projects
                        WHERE id = photos.project_id) AS best_count_per_group
                FROM photos WHERE id IN ({0})
                """.format(placeholders),
                list(photo_ids),
            ).fetchall()
            for row in rows:
                if row["decision"] == decision.value and row["decision_source"] == source:
                    continue
                connection.execute(
                    """
                    INSERT INTO manual_overrides(
                        photo_id, project_id, action_type, previous_value,
                        previous_source, new_value, analysis_profile, analysis_version,
                        photo_path, auto_reasons, focus_score, brightness, clipped_dark,
                        clipped_bright, face_count, eye_count, closed_eye_count,
                        uncertain_eye_count, subject_focus_score, face_confidence,
                        eye_open_score, eye_focus_score, object_count,
                        primary_subject_focus_score, best_score, similarity_group,
                        best_count_per_group
                    ) VALUES (
                        ?, ?, 'decision', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        row["id"],
                        row["project_id"],
                        row["decision"],
                        row["decision_source"],
                        decision.value,
                        row["analysis_profile"],
                        row["analysis_version"],
                        row["relative_path"],
                        row["auto_reasons"],
                        row["focus_score"],
                        row["brightness"],
                        row["clipped_dark"],
                        row["clipped_bright"],
                        row["face_count"],
                        row["eye_count"],
                        row["closed_eye_count"],
                        row["uncertain_eye_count"],
                        row["subject_focus_score"],
                        row["face_confidence"],
                        row["eye_open_score"],
                        row["eye_focus_score"],
                        row["object_count"],
                        row["primary_subject_focus_score"],
                        row["best_score"],
                        row["similarity_group"],
                        row["best_count_per_group"],
                    ),
                )
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
        analysis_profile: str = "safe",
        analysis_version: str = ANALYSIS_VERSION,
    ) -> None:
        """Persist analysis without overwriting an explicit manual decision."""

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE photos SET
                    analysis_status = ?,
                    analysis_profile = ?,
                    analysis_version = ?,
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
                    get_analysis_profile(analysis_profile).key,
                    analysis_version,
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
            rows = connection.execute(
                """
                SELECT id, project_id, relative_path, is_best, best_source,
                       analysis_profile, analysis_version, auto_reasons,
                       focus_score, brightness, clipped_dark, clipped_bright,
                       face_count, eye_count, closed_eye_count, uncertain_eye_count,
                       subject_focus_score, face_confidence, eye_open_score,
                       eye_focus_score, object_count, primary_subject_focus_score,
                       best_score, similarity_group,
                       (SELECT best_count_per_group FROM projects
                        WHERE id = photos.project_id) AS best_count_per_group
                FROM photos WHERE id IN ({0})
                """.format(placeholders),
                list(photo_ids),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO manual_overrides(
                        photo_id, project_id, action_type, previous_value,
                        previous_source, new_value, analysis_profile, analysis_version,
                        photo_path, auto_reasons, focus_score, brightness, clipped_dark,
                        clipped_bright, face_count, eye_count, closed_eye_count,
                        uncertain_eye_count, subject_focus_score, face_confidence,
                        eye_open_score, eye_focus_score, object_count,
                        primary_subject_focus_score, best_score, similarity_group,
                        best_count_per_group
                    ) VALUES (
                        ?, ?, 'best', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        row["id"],
                        row["project_id"],
                        "on" if row["is_best"] else "off",
                        row["best_source"],
                        "off" if row["is_best"] else "on",
                        row["analysis_profile"],
                        row["analysis_version"],
                        row["relative_path"],
                        row["auto_reasons"],
                        row["focus_score"],
                        row["brightness"],
                        row["clipped_dark"],
                        row["clipped_bright"],
                        row["face_count"],
                        row["eye_count"],
                        row["closed_eye_count"],
                        row["uncertain_eye_count"],
                        row["subject_focus_score"],
                        row["face_confidence"],
                        row["eye_open_score"],
                        row["eye_focus_score"],
                        row["object_count"],
                        row["primary_subject_focus_score"],
                        row["best_score"],
                        row["similarity_group"],
                        row["best_count_per_group"],
                    ),
                )
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

    def set_project_analysis_profile(self, project_id: int, profile: str) -> None:
        """Select a profile without silently re-running or replacing old decisions."""

        selected = get_analysis_profile(profile)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT analysis_profile FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown folder id: {0}".format(project_id))
            if row["analysis_profile"] == selected.key:
                return
            connection.execute(
                "UPDATE projects SET analysis_profile = ? WHERE id = ?",
                (selected.key, project_id),
            )
            connection.execute(
                """
                UPDATE photos SET analysis_status = 'stale'
                WHERE project_id = ? AND analysis_status != 'pending'
                """,
                (project_id,),
            )
            connection.commit()

    def count_stale_analysis(self, project_id: int) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM photos "
                "WHERE project_id = ? AND analysis_status = 'stale'",
                (project_id,),
            ).fetchone()
        return int(row["total"])

    def count_pending_analysis(self, project_id: int) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM photos "
                "WHERE project_id = ? AND analysis_status = 'pending'",
                (project_id,),
            ).fetchone()
        return int(row["total"])

    def list_manual_overrides(self, project_id: int) -> List[Dict[str, object]]:
        """Return local calibration evidence without reading or exporting image data."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, photo_id, action_type, previous_value, previous_source,
                       new_value, analysis_profile, analysis_version, photo_path,
                       auto_reasons, focus_score, brightness, clipped_dark,
                       clipped_bright, face_count, eye_count, closed_eye_count,
                       uncertain_eye_count, subject_focus_score, face_confidence,
                       eye_open_score, eye_focus_score, object_count,
                       primary_subject_focus_score, best_score, similarity_group,
                       best_count_per_group, created_at
                FROM manual_overrides
                WHERE project_id = ?
                ORDER BY id
                """,
                (project_id,),
            ).fetchall()
        overrides = []
        for row in rows:
            override = dict(row)
            try:
                override["auto_reasons"] = tuple(
                    json.loads(override["auto_reasons"] or "[]")
                )
            except (TypeError, ValueError):
                override["auto_reasons"] = ()
            overrides.append(override)
        return overrides

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

    def replace_identity_faces(
        self,
        photo_id: int,
        faces: Sequence[IdentityFaceEvidence],
    ) -> None:
        """Replace inferred faces while preserving assignments for the same regions."""

        with closing(self._connect()) as connection:
            previous = connection.execute(
                """
                SELECT f.id, f.face_index, f.bounds, f.model_name,
                       f.assignment_source, persons.display_name
                FROM face_instances AS f
                LEFT JOIN persons ON persons.id = f.person_id
                WHERE f.photo_id = ?
                """,
                (photo_id,),
            ).fetchall()
            old_bounds = {
                int(row["id"]): tuple(float(value) for value in json.loads(row["bounds"]))
                for row in previous
            }
            remaining = set(old_bounds)
            old_models = {int(row["id"]): str(row["model_name"]) for row in previous}
            reset_on_model_change = {
                int(row["id"])
                for row in previous
                if row["assignment_source"] == "automatic"
                and not str(row["display_name"] or "").strip()
            }
            if previous:
                connection.execute(
                    "UPDATE face_instances SET face_index = -id WHERE photo_id = ?",
                    (photo_id,),
                )

            for face in faces:
                matched_id = None
                matched_iou = 0.0
                for candidate_id in remaining:
                    overlap = self._bounds_iou(face.bounds, old_bounds[candidate_id])
                    if overlap > matched_iou:
                        matched_iou = overlap
                        matched_id = candidate_id
                embedding = np.asarray(face.embedding, dtype=np.float32)
                if not embedding.size:
                    continue
                values = (
                    face.face_index,
                    json.dumps(list(face.bounds), separators=(",", ":")),
                    json.dumps(
                        [list(point) for point in face.landmarks],
                        separators=(",", ":"),
                    ),
                    embedding.tobytes(),
                    int(embedding.size),
                    float(face.quality),
                    face.model_name,
                )
                if matched_id is not None and matched_iou >= 0.45:
                    connection.execute(
                        """
                        UPDATE face_instances SET
                            face_index = ?, bounds = ?, landmarks = ?, embedding = ?,
                            embedding_dimension = ?, quality = ?, model_name = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        values + (matched_id,),
                    )
                    if (
                        matched_id in reset_on_model_change
                        and old_models[matched_id] != face.model_name
                    ):
                        connection.execute(
                            """
                            UPDATE face_instances SET person_id = NULL,
                                assignment_source = 'unknown',
                                assignment_confidence = NULL
                            WHERE id = ?
                            """,
                            (matched_id,),
                        )
                    remaining.discard(matched_id)
                else:
                    connection.execute(
                        """
                        INSERT INTO face_instances(
                            photo_id, face_index, bounds, landmarks, embedding,
                            embedding_dimension, quality, model_name
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (photo_id,) + values,
                    )

            if remaining:
                placeholders = ",".join("?" for _ in remaining)
                connection.execute(
                    "DELETE FROM face_instances WHERE id IN ({0})".format(placeholders),
                    list(remaining),
                )
            connection.execute(
                """
                DELETE FROM persons
                WHERE NULLIF(TRIM(display_name), '') IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM face_instances
                      WHERE face_instances.person_id = persons.id
                  )
                """
            )
            connection.commit()

    def copy_identity_faces(self, source_photo_id: int, target_photo_id: int) -> None:
        """Copy identity evidence for an exact duplicate photograph."""

        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM face_instances WHERE photo_id = ?",
                (target_photo_id,),
            )
            connection.execute(
                """
                INSERT INTO face_instances(
                    photo_id, face_index, bounds, landmarks, embedding,
                    embedding_dimension, quality, model_name, person_id,
                    assignment_source, assignment_confidence
                )
                SELECT ?, face_index, bounds, landmarks, embedding,
                       embedding_dimension, quality, model_name, person_id,
                       assignment_source, assignment_confidence
                FROM face_instances WHERE photo_id = ?
                """,
                (target_photo_id, source_photo_id),
            )
            connection.commit()

    def list_identity_faces(self) -> List[Dict[str, object]]:
        """Return local embeddings and assignment metadata for identity grouping."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT f.id, f.photo_id, f.face_index, f.bounds, f.landmarks,
                       f.embedding, f.embedding_dimension, f.quality, f.model_name,
                       f.person_id, f.assignment_source, f.assignment_confidence,
                       p.project_id,
                       CASE WHEN NULLIF(TRIM(persons.display_name), '') IS NULL
                            THEN 0 ELSE 1 END AS person_is_named
                FROM face_instances AS f
                JOIN photos AS p ON p.id = f.photo_id
                LEFT JOIN persons ON persons.id = f.person_id
                ORDER BY f.id
                """
            ).fetchall()
        results: List[Dict[str, object]] = []
        for row in rows:
            item = dict(row)
            dimension = int(item["embedding_dimension"])
            item["embedding"] = np.frombuffer(
                item["embedding"], dtype=np.float32, count=dimension
            ).copy()
            item["bounds"] = tuple(float(value) for value in json.loads(item["bounds"]))
            item["landmarks"] = tuple(
                (float(point[0]), float(point[1]))
                for point in json.loads(item["landmarks"])
            )
            results.append(item)
        return results

    def create_person(self) -> int:
        with closing(self._connect()) as connection:
            cursor = connection.execute("INSERT INTO persons DEFAULT VALUES")
            connection.commit()
            return int(cursor.lastrowid)

    def assign_identity_face(
        self,
        face_id: int,
        person_id: int,
        confidence: float,
        source: str = "automatic",
    ) -> None:
        if source not in {"automatic", "manual"}:
            raise ValueError("Invalid identity assignment source")
        with closing(self._connect()) as connection:
            rejected = connection.execute(
                """
                SELECT 1 FROM identity_rejections
                WHERE face_id = ? AND person_id = ?
                """,
                (face_id, person_id),
            ).fetchone()
            if rejected is not None:
                return
            connection.execute(
                """
                UPDATE face_instances SET
                    person_id = ?, assignment_source = ?, assignment_confidence = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (person_id, source, float(confidence), face_id),
            )
            connection.execute(
                "UPDATE persons SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (person_id,),
            )
            connection.commit()

    def person_has_photo(
        self,
        person_id: int,
        photo_id: int,
        excluding_face_id: Optional[int] = None,
    ) -> bool:
        parameters: List[object] = [person_id, photo_id]
        exclusion = ""
        if excluding_face_id is not None:
            exclusion = " AND id != ?"
            parameters.append(excluding_face_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM face_instances
                WHERE person_id = ? AND photo_id = ?{0}
                LIMIT 1
                """.format(exclusion),
                parameters,
            ).fetchone()
        return row is not None

    def rejected_person_ids(self, face_id: int) -> Tuple[int, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT person_id FROM identity_rejections WHERE face_id = ?",
                (face_id,),
            ).fetchall()
        return tuple(int(row["person_id"]) for row in rows)

    def rename_person(self, person_id: int, display_name: str) -> None:
        name = display_name.strip()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE persons SET display_name = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (name or None, person_id),
            )
            connection.commit()

    def reject_photo_person(self, photo_id: int, person_id: int) -> int:
        """Remove one person's face assignments from a photo and remember it."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id FROM face_instances
                WHERE photo_id = ? AND person_id = ?
                """,
                (photo_id, person_id),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO identity_rejections(face_id, person_id)
                    VALUES (?, ?)
                    """,
                    (row["id"], person_id),
                )
            connection.execute(
                """
                UPDATE face_instances SET
                    person_id = NULL, assignment_source = 'manual',
                    assignment_confidence = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE photo_id = ? AND person_id = ?
                """,
                (photo_id, person_id),
            )
            connection.commit()
        return len(rows)

    def merge_persons(self, target_person_id: int, source_person_id: int) -> None:
        if target_person_id == source_person_id:
            return
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE face_instances SET person_id = ?, assignment_source = 'manual',
                    updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
                  AND id NOT IN (
                      SELECT face_id FROM identity_rejections WHERE person_id = ?
                  )
                """,
                (target_person_id, source_person_id, target_person_id),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO identity_rejections(face_id, person_id)
                SELECT face_id, ? FROM identity_rejections
                WHERE person_id = ?
                """,
                (target_person_id, source_person_id),
            )
            connection.execute("DELETE FROM persons WHERE id = ?", (source_person_id,))
            connection.execute(
                "UPDATE persons SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (target_person_id,),
            )
            connection.commit()

    def merge_automatic_person(
        self,
        target_person_id: int,
        source_person_id: int,
        confidence: float,
    ) -> None:
        """Merge an entirely automatic fragment without creating manual truth."""

        if target_person_id == source_person_id:
            return
        with closing(self._connect()) as connection:
            source = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(assignment_source != 'automatic') AS protected
                FROM face_instances WHERE person_id = ?
                """,
                (source_person_id,),
            ).fetchone()
            if source is None or not int(source["total"] or 0):
                return
            if int(source["protected"] or 0):
                return
            connection.execute(
                """
                UPDATE face_instances SET person_id = ?,
                    assignment_confidence = ?, updated_at = CURRENT_TIMESTAMP
                WHERE person_id = ?
                  AND id NOT IN (
                      SELECT face_id FROM identity_rejections WHERE person_id = ?
                  )
                """,
                (
                    target_person_id,
                    max(0.0, min(1.0, float(confidence))),
                    source_person_id,
                    target_person_id,
                ),
            )
            connection.execute("DELETE FROM persons WHERE id = ?", (source_person_id,))
            connection.execute(
                "UPDATE persons SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (target_person_id,),
            )
            connection.commit()

    def list_people(self, project_id: Optional[int] = None) -> List[PersonIdentity]:
        parameters: List[object] = []
        project_clause = ""
        if project_id is not None:
            project_clause = " AND ph.project_id = ?"
            parameters.append(project_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT persons.id, persons.display_name,
                       COUNT(fi.id) AS face_count,
                       COUNT(DISTINCT fi.photo_id) AS photo_count
                FROM persons
                JOIN face_instances AS fi ON fi.person_id = persons.id
                JOIN photos AS ph ON ph.id = fi.photo_id
                WHERE 1 = 1{0}
                GROUP BY persons.id
                ORDER BY COALESCE(NULLIF(persons.display_name, ''),
                                  printf('Person %08d', persons.id)) COLLATE NOCASE
                """.format(project_clause),
                parameters,
            ).fetchall()
            cover_rows = connection.execute(
                """
                SELECT fi.person_id, fi.bounds, ph.relative_path,
                       ph.preview_relative_path, ph.cached_preview_path,
                       projects.root_path
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                JOIN projects ON projects.id = ph.project_id
                WHERE fi.person_id IS NOT NULL{0}
                ORDER BY fi.person_id, fi.quality DESC, fi.id
                """.format(project_clause),
                parameters,
            ).fetchall()
            covers = {}
            for cover in cover_rows:
                covers.setdefault(int(cover["person_id"]), cover)

            people: List[PersonIdentity] = []
            for row in rows:
                cover = covers.get(int(row["id"]))
                cover_path = None
                cover_bounds = None
                if cover is not None:
                    if cover["cached_preview_path"]:
                        cover_path = str(cover["cached_preview_path"])
                    else:
                        relative = cover["preview_relative_path"] or cover["relative_path"]
                        cover_path = str(Path(cover["root_path"]) / Path(relative))
                    cover_bounds = tuple(
                        float(value) for value in json.loads(cover["bounds"])
                    )
                explicit_name = str(row["display_name"] or "").strip()
                people.append(
                    PersonIdentity(
                        id=int(row["id"]),
                        display_name=explicit_name or "Person {0}".format(row["id"]),
                        photo_count=int(row["photo_count"]),
                        face_count=int(row["face_count"]),
                        cover_photo_path=cover_path,
                        cover_bounds=cover_bounds,
                        is_named=bool(explicit_name),
                    )
                )
        return people

    def list_person_folders(self, person_id: int) -> List[PersonFolder]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT projects.id, projects.name, COUNT(DISTINCT ph.id) AS photo_count
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                JOIN projects ON projects.id = ph.project_id
                WHERE fi.person_id = ?
                GROUP BY projects.id
                ORDER BY projects.name COLLATE NOCASE
                """,
                (person_id,),
            ).fetchall()
        return [
            PersonFolder(int(row["id"]), str(row["name"]), int(row["photo_count"]))
            for row in rows
        ]

    def identity_unknown_count(self, project_id: Optional[int] = None) -> int:
        parameters: List[object] = []
        project_clause = ""
        if project_id is not None:
            project_clause = " AND ph.project_id = ?"
            parameters.append(project_id)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                WHERE fi.person_id IS NULL{0}
                """.format(project_clause),
                parameters,
            ).fetchone()
        return int(row["total"])

    def list_photos_for_person(
        self,
        person_id: Optional[int],
        project_id: Optional[int] = None,
        view: PhotoView = PhotoView.ALL,
    ) -> List[Photo]:
        """List unique photos matching both an identity and decision view."""

        with closing(self._connect()) as connection:
            parameters: List[object] = []
            identity_clause = "fi.person_id IS NULL"
            if person_id is not None:
                identity_clause = "fi.person_id = ?"
                parameters.append(person_id)
            project_clause = ""
            if project_id is not None:
                project_clause = " AND ph.project_id = ?"
                parameters.append(project_id)
            view_clause = ""
            if view == PhotoView.BEST:
                view_clause = " AND ph.is_best = 1"
            elif view != PhotoView.ALL:
                view_clause = " AND ph.decision = ?"
                parameters.append(view.value)
            rows = connection.execute(
                """
                SELECT DISTINCT ph.*, projects.root_path
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                JOIN projects ON projects.id = ph.project_id
                WHERE {0}{1}{2}
                ORDER BY ph.captured_at IS NULL, ph.captured_at,
                         ph.relative_path COLLATE NOCASE
                """.format(identity_clause, project_clause, view_clause),
                parameters,
            ).fetchall()
        return [self._photo_from_row(row) for row in rows]

    def identity_counts(
        self,
        person_id: Optional[int],
        project_id: int,
    ) -> Dict[str, int]:
        """Count each decision view inside one folder-scoped identity filter."""

        counts = {view.value: 0 for view in PhotoView}
        identity_clause = "fi.person_id IS NULL"
        parameters: List[object] = []
        if person_id is not None:
            identity_clause = "fi.person_id = ?"
            parameters.append(person_id)
        parameters.append(project_id)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT ph.decision, COUNT(DISTINCT ph.id) AS total
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                WHERE {0} AND ph.project_id = ?
                GROUP BY ph.decision
                """.format(identity_clause),
                parameters,
            ).fetchall()
            best = connection.execute(
                """
                SELECT COUNT(DISTINCT ph.id) AS total
                FROM face_instances AS fi
                JOIN photos AS ph ON ph.id = fi.photo_id
                WHERE {0} AND ph.project_id = ? AND ph.is_best = 1
                """.format(identity_clause),
                parameters,
            ).fetchone()
        for row in rows:
            counts[str(row["decision"])] = int(row["total"])
            counts[PhotoView.ALL.value] += int(row["total"])
        counts[PhotoView.BEST.value] = int(best["total"])
        return counts

    def person_ids_for_photo(self, photo_id: int) -> Tuple[int, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT person_id FROM face_instances
                WHERE photo_id = ? AND person_id IS NOT NULL
                ORDER BY person_id
                """,
                (photo_id,),
            ).fetchall()
        return tuple(int(row["person_id"]) for row in rows)

    def face_indices_for_person(
        self,
        person_id: int,
        photo_ids: Optional[Sequence[int]] = None,
    ) -> Dict[int, Tuple[int, ...]]:
        """Return the detected face numbers assigned to one person by photograph."""

        parameters: List[object] = [person_id]
        photo_clause = ""
        if photo_ids is not None:
            normalized_ids = tuple(dict.fromkeys(int(photo_id) for photo_id in photo_ids))
            if not normalized_ids:
                return {}
            photo_clause = " AND photo_id IN ({0})".format(
                ",".join("?" for _ in normalized_ids)
            )
            parameters.extend(normalized_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT photo_id, face_index FROM face_instances
                WHERE person_id = ?{0}
                ORDER BY photo_id, face_index
                """.format(photo_clause),
                parameters,
            ).fetchall()
        grouped: Dict[int, List[int]] = {}
        for row in rows:
            grouped.setdefault(int(row["photo_id"]), []).append(int(row["face_index"]))
        return {photo_id: tuple(indices) for photo_id, indices in grouped.items()}

    def people_for_photo(self, photo_id: int) -> List[Tuple[int, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT persons.id, persons.display_name
                FROM face_instances
                JOIN persons ON persons.id = face_instances.person_id
                WHERE face_instances.photo_id = ?
                ORDER BY COALESCE(NULLIF(persons.display_name, ''),
                                  printf('Person %08d', persons.id)) COLLATE NOCASE
                """,
                (photo_id,),
            ).fetchall()
        return [
            (
                int(row["id"]),
                str(row["display_name"] or "").strip()
                or "Person {0}".format(row["id"]),
            )
            for row in rows
        ]

    def reset_identity_data(self) -> Tuple[int, int]:
        """Delete local biometric embeddings, identities, and correction constraints."""

        with closing(self._connect()) as connection:
            face_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS total FROM face_instances"
                ).fetchone()["total"]
            )
            person_count = int(
                connection.execute(
                    "SELECT COUNT(*) AS total FROM persons"
                ).fetchone()["total"]
            )
            connection.execute("DELETE FROM face_instances")
            connection.execute("DELETE FROM persons")
            connection.execute("DELETE FROM identity_metadata")
            connection.commit()
        return person_count, face_count

    def prepare_identity_regroup(self, grouping_version: str) -> bool:
        """Clear obsolete unnamed automatic groups once per grouping version."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value FROM identity_metadata WHERE key = 'grouping_version'"
            ).fetchone()
            if row is not None and row["value"] == grouping_version:
                return False
            connection.execute(
                """
                UPDATE face_instances SET
                    person_id = NULL, assignment_source = 'unknown',
                    assignment_confidence = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE assignment_source = 'automatic'
                  AND person_id IN (
                      SELECT id FROM persons
                      WHERE NULLIF(TRIM(display_name), '') IS NULL
                  )
                """
            )
            connection.execute(
                """
                DELETE FROM persons
                WHERE NULLIF(TRIM(display_name), '') IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM face_instances
                      WHERE face_instances.person_id = persons.id
                  )
                """
            )
            connection.commit()
        return True

    def identity_regroup_needed(self, grouping_version: str) -> bool:
        with closing(self._connect()) as connection:
            has_faces = connection.execute(
                "SELECT 1 FROM face_instances LIMIT 1"
            ).fetchone()
            if has_faces is None:
                return False
            row = connection.execute(
                "SELECT value FROM identity_metadata WHERE key = 'grouping_version'"
            ).fetchone()
        return row is None or row["value"] != grouping_version

    def finish_identity_regroup(self, grouping_version: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO identity_metadata(key, value)
                VALUES ('grouping_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (grouping_version,),
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
        return Project(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            analysis_profile=row["analysis_profile"],
            best_count_per_group=row["best_count_per_group"],
        )

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        return bin(int(left, 16) ^ int(right, 16)).count("1")

    @staticmethod
    def _bounds_iou(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        left_x, left_y, left_width, left_height = left
        right_x, right_y, right_width, right_height = right
        x1 = max(left_x, right_x)
        y1 = max(left_y, right_y)
        x2 = min(left_x + left_width, right_x + right_width)
        y2 = min(left_y + left_height, right_y + right_height)
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = left_width * left_height + right_width * right_height - intersection
        return intersection / union if union > 0.0 else 0.0

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
            analysis_profile=row["analysis_profile"],
            analysis_version=row["analysis_version"],
            focus_score=row["focus_score"],
            brightness=row["brightness"],
            clipped_dark=row["clipped_dark"],
            clipped_bright=row["clipped_bright"],
            auto_reasons=tuple(json.loads(row["auto_reasons"] or "[]")),
            duplicate_of_id=row["duplicate_of_id"],
            best_source=row["best_source"],
            perceptual_hash=row["perceptual_hash"],
            content_hash=row["content_hash"],
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
            motion_count=row["motion_count"],
            burst_id=row["burst_id"],
            live_photo_id=row["live_photo_id"],
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
                            "visibility_reason": eye.visibility_reason,
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
                            visibility_reason=(
                                str(eye["visibility_reason"])
                                if eye.get("visibility_reason")
                                else None
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
