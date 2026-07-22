from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from photochoose.database import PhotoDatabase
from photochoose.models import (
    Decision,
    EyeEvidence,
    FaceEvidence,
    ObjectEvidence,
    PhotoView,
)


class PhotoDatabaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shoot = self.root / "Wedding"
        self.shoot.mkdir()
        (self.shoot / "Camera A").mkdir()
        self.database = PhotoDatabase(self.root / "state" / "photochoose.sqlite3")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_photo(self, relative_path: str) -> Path:
        path = self.shoot / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test image placeholder")
        return path

    def test_project_is_added_only_once(self) -> None:
        first = self.database.add_project(self.shoot)
        second = self.database.add_project(self.shoot)

        self.assertEqual(first.id, second.id)
        self.assertEqual([first], self.database.list_projects())

    def test_scan_discovers_supported_images_recursively(self) -> None:
        self._write_photo("one.JPG")
        self._write_photo("Camera A/two.jpeg")
        self._write_photo("notes.txt")
        project = self.database.add_project(self.shoot)

        count = self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)

        self.assertEqual(2, count)
        self.assertEqual(
            ["Camera A/two.jpeg", "one.JPG"],
            [photo.relative_path for photo in photos],
        )

    def test_decisions_and_best_flag_are_independent(self) -> None:
        self._write_photo("one.jpg")
        self._write_photo("two.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)

        self.database.set_decision([photos[0].id], Decision.KEEP)
        self.database.toggle_best([photos[0].id])

        keep = self.database.list_photos(project.id, PhotoView.KEEP)
        best = self.database.list_photos(project.id, PhotoView.BEST)
        self.assertEqual([photos[0].id], [photo.id for photo in keep])
        self.assertEqual([photos[0].id], [photo.id for photo in best])
        self.assertTrue(keep[0].is_best)
        self.assertEqual("manual", keep[0].best_source)

    def test_rescan_preserves_decisions_and_removes_missing_records(self) -> None:
        first_path = self._write_photo("one.jpg")
        self._write_photo("two.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        first = next(photo for photo in photos if photo.relative_path == "one.jpg")
        self.database.set_decision([first.id], Decision.REJECT)

        first_path.unlink()
        self._write_photo("three.jpg")
        self.database.scan_project(project.id)

        remaining = self.database.list_photos(project.id, PhotoView.ALL)
        self.assertEqual(["three.jpg", "two.jpg"], [photo.relative_path for photo in remaining])
        self.assertEqual([], self.database.list_photos(project.id, PhotoView.REJECT))

    def test_counts_include_unreviewed_in_all(self) -> None:
        self._write_photo("one.jpg")
        self._write_photo("two.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        self.database.set_decision([photos[0].id], Decision.REVIEW)
        self.database.toggle_best([photos[1].id])

        counts = self.database.counts(project.id)

        self.assertEqual(2, counts[PhotoView.ALL.value])
        self.assertEqual(1, counts[PhotoView.REVIEW.value])
        self.assertEqual(1, counts[PhotoView.BEST.value])

    def test_existing_database_receives_analysis_columns(self) -> None:
        old_path = self.root / "old.sqlite3"
        connection = sqlite3.connect(str(old_path))
        connection.executescript(
            """
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_opened_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                relative_path TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_ns INTEGER NOT NULL,
                decision TEXT NOT NULL DEFAULT 'unreviewed',
                is_best INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, relative_path)
            );
            """
        )
        connection.close()

        PhotoDatabase(old_path)

        migrated = sqlite3.connect(str(old_path))
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(photos)")}
        project_columns = {row[1] for row in migrated.execute("PRAGMA table_info(projects)")}
        migrated.close()
        self.assertIn("analysis_status", columns)
        self.assertIn("auto_reasons", columns)
        self.assertIn("decision_source", columns)
        self.assertIn("best_source", columns)
        self.assertIn("perceptual_hash", columns)
        self.assertIn("face_count", columns)
        self.assertIn("closed_eye_count", columns)
        self.assertIn("uncertain_eye_count", columns)
        self.assertIn("subject_model", columns)
        self.assertIn("face_confidence", columns)
        self.assertIn("eye_open_score", columns)
        self.assertIn("eye_focus_score", columns)
        self.assertIn("subject_regions", columns)
        self.assertIn("object_count", columns)
        self.assertIn("object_model", columns)
        self.assertIn("primary_subject_focus_score", columns)
        self.assertIn("subject_objects", columns)
        self.assertIn("similarity_group", columns)
        self.assertIn("group_reason", columns)
        self.assertIn("captured_at", columns)
        self.assertIn("camera_model", columns)
        self.assertIn("exposure_seconds", columns)
        self.assertIn("preview_relative_path", columns)
        self.assertIn("asset_paths", columns)
        self.assertIn("raw_count", columns)
        self.assertIn("asset_key", columns)
        self.assertIn("is_open", project_columns)

    def test_subject_evidence_round_trips_through_database(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        eye = EyeEvidence(
            state="open",
            aperture=0.24,
            openness_score=1.0,
            focus_score=321.5,
            landmarks=((0.2, 0.3), (0.25, 0.31)),
        )
        face = FaceEvidence(
            bounds=(0.1, 0.15, 0.4, 0.5),
            confidence=0.97,
            focus_score=210.0,
            important=True,
            eyes=(eye,),
        )

        self.database.apply_subject_analysis(
            photo.id,
            "0123456789abcdef",
            1,
            1,
            0,
            0,
            210.0,
            "test-model",
            0.97,
            1.0,
            321.5,
            (face,),
        )

        loaded = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.assertEqual((face,), loaded.subject_regions)
        self.assertEqual(321.5, loaded.eye_focus_score)
        self.assertEqual(1, loaded.important_face_count)

    def test_object_evidence_round_trips_through_database(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        detected = ObjectEvidence(
            label="dog",
            bounds=(0.1, 0.2, 0.5, 0.6),
            confidence=0.91,
            focus_score=142.5,
            importance="primary",
        )

        self.database.apply_object_analysis(
            photo.id,
            1,
            "nanodet-opencv",
            142.5,
            (detected,),
        )

        loaded = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.assertEqual((detected,), loaded.subject_objects)
        self.assertEqual(1, loaded.object_count)
        self.assertEqual(1, loaded.primary_object_count)
        self.assertEqual("nanodet-opencv", loaded.object_model)

    def test_closing_and_reopening_project_preserves_decisions(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.database.set_decision([photo.id], Decision.KEEP)

        self.database.close_project(project.id)

        self.assertEqual([], self.database.list_projects())
        reopened = self.database.add_project(self.shoot)
        self.assertEqual(project.id, reopened.id)
        preserved = self.database.list_photos(reopened.id, PhotoView.ALL)[0]
        self.assertEqual(Decision.KEEP, preserved.decision)
        self.assertEqual("manual", preserved.decision_source)


if __name__ == "__main__":
    unittest.main()
