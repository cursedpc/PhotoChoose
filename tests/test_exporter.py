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
        self.database = PhotoDatabase(self.root / "photochoose.sqlite3")
        self.project = self.database.add_project(self.shoot)
        self.database.scan_project(self.project.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_export_copies_categories_and_preserves_relative_paths(self) -> None:
        photos = {
            photo.filename: photo
            for photo in self.database.list_photos(self.project.id, PhotoView.ALL)
        }
        keep = photos["keep.jpg"]
        best = photos["best.jpg"]
        self.database.set_decision([keep.id], Decision.KEEP)
        self.database.toggle_best([keep.id, best.id])
        destination = self.root / "Export"
        progress = []

        result = ProjectExporter(self.database).export_keep_and_best(
            self.project.id,
            destination,
            lambda complete, total: progress.append((complete, total)),
        )

        self.assertEqual(3, result.copied)
        self.assertEqual(0, result.skipped)
        self.assertEqual((3, 3), progress[-1])
        self.assertEqual(
            b"keep bytes", (destination / "Keep" / "Camera A" / "keep.jpg").read_bytes()
        )
        self.assertEqual(
            b"keep bytes", (destination / "Best" / "Camera A" / "keep.jpg").read_bytes()
        )
        self.assertEqual(b"best bytes", (destination / "Best" / "best.jpg").read_bytes())

        existing = destination / "Best" / "best.jpg"
        existing.write_bytes(b"do not overwrite")
        second = ProjectExporter(self.database).export_keep_and_best(self.project.id, destination)
        self.assertEqual(0, second.copied)
        self.assertEqual(3, second.skipped)
        self.assertEqual(b"do not overwrite", existing.read_bytes())

    def test_export_rejects_destination_inside_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the source"):
            ProjectExporter(self.database).export_keep_and_best(
                self.project.id, self.shoot / "Export"
            )


if __name__ == "__main__":
    unittest.main()
