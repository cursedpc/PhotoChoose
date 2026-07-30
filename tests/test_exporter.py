from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photochoose.database import PhotoDatabase
from photochoose.exporter import ProjectExporter
from photochoose.models import Decision, PhotoView


class ProjectExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.shoot = self.root / "Shoot"
        (self.shoot / "Camera A").mkdir(parents=True)
        (self.shoot / "Camera A" / "keep.jpg").write_bytes(b"keep bytes")
        (self.shoot / "best.jpg").write_bytes(b"best bytes")
        (self.shoot / "other.jpg").write_bytes(b"other bytes")
        self.database = PhotoDatabase(self.root / "photochoose.sqlite3")
        self.project = self.database.add_project(self.shoot)
        self.database.scan_project(self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_keep_and_best_exports_are_separate_and_preserve_paths(self) -> None:
        photos = {
            photo.filename: photo
            for photo in self.database.list_photos(self.project.id, PhotoView.ALL)
        }
        keep = photos["keep.jpg"]
        best = photos["best.jpg"]
        self.database.set_decision([keep.id], Decision.KEEP)
        self.database.toggle_best([keep.id, best.id])
        keep_destination = self.root / "Keep Export"
        best_destination = self.root / "Best Export"
        keep_progress = []
        best_progress = []

        keep_result = ProjectExporter(self.database).export_keep(
            self.project.id,
            keep_destination,
            lambda complete, total: keep_progress.append((complete, total)),
        )
        best_result = ProjectExporter(self.database).export_best(
            self.project.id,
            best_destination,
            lambda complete, total: best_progress.append((complete, total)),
        )

        self.assertEqual(1, keep_result.copied)
        self.assertEqual(2, best_result.copied)
        self.assertEqual((1, 1), keep_progress[-1])
        self.assertEqual((2, 2), best_progress[-1])
        self.assertEqual(
            b"keep bytes", (keep_destination / "Camera A" / "keep.jpg").read_bytes()
        )
        self.assertEqual(
            b"keep bytes", (best_destination / "Camera A" / "keep.jpg").read_bytes()
        )
        self.assertEqual(b"best bytes", (best_destination / "best.jpg").read_bytes())
        self.assertFalse((keep_destination / "best.jpg").exists())

        existing = best_destination / "best.jpg"
        existing.write_bytes(b"do not overwrite")
        second = ProjectExporter(self.database).export_best(
            self.project.id, best_destination
        )
        self.assertEqual(0, second.copied)
        self.assertEqual(2, second.skipped)
        self.assertEqual(b"do not overwrite", existing.read_bytes())

    def test_export_rejects_destination_inside_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the source"):
            ProjectExporter(self.database).export_keep(
                self.project.id, self.shoot / "Export"
            )

    def test_export_selected_ignores_decisions_and_preserves_paths(self) -> None:
        photos = {
            photo.filename: photo
            for photo in self.database.list_photos(self.project.id, PhotoView.ALL)
        }
        selected = (photos["keep.jpg"], photos["best.jpg"])
        self.database.set_decision([selected[1].id], Decision.REJECT)
        destination = self.root / "Selected"
        progress = []

        result = ProjectExporter(self.database).export_selected(
            self.project.id,
            [photo.id for photo in selected],
            destination,
            lambda complete, total: progress.append((complete, total)),
        )

        self.assertEqual(2, result.copied)
        self.assertEqual((2, 2), progress[-1])
        self.assertEqual(
            b"keep bytes", (destination / "Camera A" / "keep.jpg").read_bytes()
        )
        self.assertEqual(b"best bytes", (destination / "best.jpg").read_bytes())
        self.assertFalse((destination / "other.jpg").exists())
        self.assertFalse((destination / "Keep").exists())


if __name__ == "__main__":
    unittest.main()
