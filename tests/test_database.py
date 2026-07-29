from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from photochoose.analysis_profiles import ANALYSIS_VERSION
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

    def test_same_filename_remains_distinct_by_folder_and_relative_path(self) -> None:
        self._write_photo("Camera A/IMG_0001.jpg")
        self._write_photo("Camera B/IMG_0001.jpg")
        second_shoot = self.root / "Portraits"
        second_shoot.mkdir()
        (second_shoot / "IMG_0001.jpg").write_bytes(b"other image placeholder")

        first_project = self.database.add_project(self.shoot)
        second_project = self.database.add_project(second_shoot)
        self.database.scan_project(first_project.id)
        self.database.scan_project(second_project.id)

        first_photos = self.database.list_photos(first_project.id, PhotoView.ALL)
        second_photos = self.database.list_photos(second_project.id, PhotoView.ALL)
        self.assertEqual(
            ["Camera A/IMG_0001.jpg", "Camera B/IMG_0001.jpg"],
            [photo.relative_path for photo in first_photos],
        )
        self.assertEqual(["IMG_0001.jpg"], [photo.relative_path for photo in second_photos])
        self.assertEqual(3, len({photo.id for photo in first_photos + second_photos}))

    def test_scan_reports_logical_asset_progress(self) -> None:
        self._write_photo("one.jpg")
        self._write_photo("Camera A/two.jpg")
        project = self.database.add_project(self.shoot)
        progress = []

        self.database.scan_project(
            project.id,
            lambda complete, total: progress.append((complete, total)),
        )

        self.assertEqual((0, 2), progress[0])
        self.assertEqual((2, 2), progress[-1])
        self.assertEqual([(0, 2), (1, 2), (2, 2)], progress)

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
            CREATE TABLE manual_overrides (
                id INTEGER PRIMARY KEY,
                photo_id INTEGER NOT NULL,
                project_id INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                previous_value TEXT NOT NULL,
                previous_source TEXT NOT NULL,
                new_value TEXT NOT NULL,
                analysis_profile TEXT,
                analysis_version TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        self.assertIn("sidecar_count", columns)
        self.assertIn("motion_count", columns)
        self.assertIn("burst_id", columns)
        self.assertIn("live_photo_id", columns)
        self.assertIn("asset_key", columns)
        self.assertIn("is_open", project_columns)
        self.assertIn("analysis_profile", columns)
        self.assertIn("analysis_version", columns)
        self.assertIn("analysis_profile", project_columns)
        self.assertIn("best_count_per_group", project_columns)
        migrated = sqlite3.connect(str(old_path))
        tables = {
            row[0]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        override_columns = {
            row[1]
            for row in migrated.execute("PRAGMA table_info(manual_overrides)")
        }
        migrated.close()
        self.assertIn("manual_overrides", tables)
        self.assertIn("auto_reasons", override_columns)
        self.assertIn("focus_score", override_columns)
        self.assertIn("eye_open_score", override_columns)
        self.assertIn("primary_subject_focus_score", override_columns)
        self.assertIn("best_count_per_group", override_columns)

    def test_profile_change_is_explicit_and_manual_overrides_are_local(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.database.apply_analysis(
            photo.id,
            Decision.REVIEW,
            18.0,
            0.5,
            0.0,
            0.0,
            "hash",
            ("Possibly soft",),
            None,
        )

        self.database.set_decision([photo.id], Decision.KEEP)
        self.database.toggle_best([photo.id])
        self.database.set_project_analysis_profile(project.id, "balanced")

        updated_project = self.database.get_project(project.id)
        stale = self.database.list_photos(project.id, PhotoView.ALL)[0]
        overrides = self.database.list_manual_overrides(project.id)
        self.assertEqual("balanced", updated_project.analysis_profile)
        self.assertEqual("stale", stale.analysis_status)
        self.assertEqual(Decision.KEEP, stale.decision)
        self.assertEqual("manual", stale.decision_source)
        self.assertTrue(stale.is_best)
        self.assertEqual(["decision", "best"], [item["action_type"] for item in overrides])
        self.assertEqual("automatic", overrides[0]["previous_source"])
        self.assertEqual("safe", overrides[0]["analysis_profile"])
        self.assertEqual(ANALYSIS_VERSION, overrides[0]["analysis_version"])
        self.assertEqual("one.jpg", overrides[0]["photo_path"])
        self.assertEqual(("Possibly soft",), overrides[0]["auto_reasons"])
        self.assertEqual(18.0, overrides[0]["focus_score"])

        self.database.apply_analysis(
            photo.id,
            Decision.REJECT,
            18.0,
            0.5,
            0.0,
            0.0,
            "hash",
            ("Strict recommendation",),
            None,
            analysis_profile="balanced",
        )
        reanalyzed = self.database.list_photos(project.id, PhotoView.ALL)[0]
        preserved_snapshot = self.database.list_manual_overrides(project.id)[0]
        self.assertEqual("complete", reanalyzed.analysis_status)
        self.assertEqual("balanced", reanalyzed.analysis_profile)
        self.assertEqual(Decision.KEEP, reanalyzed.decision)
        self.assertEqual("manual", reanalyzed.decision_source)
        self.assertEqual(("Possibly soft",), preserved_snapshot["auto_reasons"])
        self.assertEqual(18.0, preserved_snapshot["focus_score"])

    def test_best_count_selects_diverse_frames_and_preserves_manual_best(self) -> None:
        for filename in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
            self._write_photo(filename)
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photos = self.database.list_photos(project.id, PhotoView.ALL)
        hashes = {
            "a.jpg": "0000000000000000",
            "b.jpg": "0000000000000001",
            "c.jpg": "00000000000003ff",
            "d.jpg": "0000000003ff0000",
        }
        scores = {"a.jpg": 100.0, "b.jpg": 90.0, "c.jpg": 80.0, "d.jpg": 70.0}
        by_name = {photo.filename: photo for photo in photos}
        for photo in photos:
            self.database.apply_analysis(
                photo.id,
                Decision.KEEP,
                100.0,
                0.5,
                0.0,
                0.0,
                photo.filename,
                ("Passed",),
                None,
            )
            self.database.apply_subject_analysis(
                photo.id,
                hashes[photo.filename],
                0,
                0,
                0,
                0,
                None,
                "none",
                None,
                None,
            )
            self.database.set_similarity_group(
                photo.id, "B0001", scores[photo.filename], "Test burst"
            )

        selected = self.database.set_project_best_count(project.id, 3)
        first_pass = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual(3, selected)
        self.assertEqual(3, self.database.get_project(project.id).best_count_per_group)
        self.assertEqual(
            {"a.jpg", "c.jpg", "d.jpg"},
            {name for name, photo in first_pass.items() if photo.is_best},
        )

        self.database.toggle_best([by_name["b.jpg"].id])
        after_manual = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual(
            {"b.jpg", "c.jpg", "d.jpg"},
            {name for name, photo in after_manual.items() if photo.is_best},
        )
        selected = self.database.set_project_best_count(project.id, 2)
        manual_pass = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual(1, selected)
        self.assertEqual(
            {"b.jpg", "c.jpg"},
            {name for name, photo in manual_pass.items() if photo.is_best},
        )
        self.assertEqual("manual", manual_pass["b.jpg"].best_source)
        self.assertEqual("automatic", manual_pass["c.jpg"].best_source)

        self.database.set_decision([by_name["c.jpg"].id], Decision.REJECT)
        after_reject = {
            photo.filename: photo
            for photo in self.database.list_photos(project.id, PhotoView.ALL)
        }
        self.assertEqual(
            {"b.jpg", "d.jpg"},
            {name for name, photo in after_reject.items() if photo.is_best},
        )
        self.assertEqual("automatic", after_reject["d.jpg"].best_source)

        with self.assertRaises(ValueError):
            self.database.set_project_best_count(project.id, 0)
        with self.assertRaises(ValueError):
            self.database.set_project_best_count(project.id, 6)

    def test_subject_evidence_round_trips_through_database(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        eye = EyeEvidence(
            state="obscured",
            aperture=0.24,
            openness_score=None,
            focus_score=321.5,
            landmarks=((0.2, 0.3), (0.25, 0.31)),
            visibility_reason="Strong glare prevents verification",
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
            0,
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
        self.assertEqual(1, loaded.obscured_eye_count)
        self.assertEqual(
            ("Strong glare prevents verification",),
            loaded.eye_visibility_reasons,
        )

    def test_outdated_analysis_is_marked_stale_without_changing_decision(self) -> None:
        self._write_photo("one.jpg")
        project = self.database.add_project(self.shoot)
        self.database.scan_project(project.id)
        photo = self.database.list_photos(project.id, PhotoView.ALL)[0]
        self.database.apply_analysis(
            photo.id,
            Decision.REVIEW,
            20.0,
            0.5,
            0.0,
            0.0,
            "hash",
            ("Possibly soft",),
            None,
        )
        self.database.set_decision([photo.id], Decision.KEEP)
        connection = sqlite3.connect(str(self.database.path))
        connection.execute(
            "UPDATE photos SET analysis_status = 'complete', analysis_version = 'legacy-v1'"
        )
        connection.commit()
        connection.close()

        migrated = PhotoDatabase(self.database.path)
        loaded = migrated.list_photos(project.id, PhotoView.ALL)[0]

        self.assertEqual("stale", loaded.analysis_status)
        self.assertEqual(Decision.KEEP, loaded.decision)
        self.assertEqual("manual", loaded.decision_source)

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
