from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from photochoose.analyzer import BaselineAnalyzer
from photochoose.database import PhotoDatabase
from photochoose.exporter import ProjectExporter
from photochoose.models import Decision, PhotoView
from photochoose.raw_preview import extract_embedded_jpeg


def _jpeg(width: int = 1200, height: int = 800, value: int = 128) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise AssertionError("Could not encode JPEG fixture")
    return encoded.tobytes()


class RawAssetTests(unittest.TestCase):
    def test_raw_jpeg_xmp_are_one_photo_and_export_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            (shoot / "frame.cr3").write_bytes(b"private raw bytes")
            (shoot / "frame.jpg").write_bytes(_jpeg())
            (shoot / "frame.cr3.xmp").write_text("<x:xmpmeta />", encoding="utf-8")
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)

            count = database.scan_project(project.id)
            photos = database.list_photos(project.id, PhotoView.ALL)

            self.assertEqual(1, count)
            self.assertEqual(1, len(photos))
            photo = photos[0]
            self.assertEqual("frame.jpg", photo.relative_path)
            self.assertEqual(
                ("frame.cr3", "frame.jpg", "frame.cr3.xmp"), photo.asset_paths
            )
            self.assertEqual(1, photo.raw_count)
            self.assertEqual(1, photo.sidecar_count)
            self.assertEqual((shoot / "frame.jpg").resolve(), Path(photo.display_path))

            BaselineAnalyzer(database).analyze_project(project.id)
            analyzed = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertNotEqual("preview-unavailable", analyzed.analysis_status)
            database.set_decision([analyzed.id], Decision.KEEP)
            destination = root / "Export"

            result = ProjectExporter(database).export_keep(
                project.id, destination
            )

            self.assertEqual(3, result.copied)
            self.assertEqual(b"private raw bytes", (destination / "frame.cr3").read_bytes())
            self.assertTrue((destination / "frame.jpg").is_file())
            self.assertEqual(
                "<x:xmpmeta />",
                (destination / "frame.cr3.xmp").read_text(encoding="utf-8"),
            )

    def test_raw_only_uses_largest_embedded_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            raw = shoot / "frame.nef"
            small = _jpeg(400, 320, 90)
            large = _jpeg(1400, 900, 140)
            raw.write_bytes(b"RAW-HEADER" + small + b"MIDDLE" + large + b"RAW-END")

            preview = extract_embedded_jpeg(raw, root / "cache")

            self.assertIsNotNone(preview)
            decoded = cv2.imread(str(preview))
            self.assertEqual((900, 1400), decoded.shape[:2])

            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)
            photo = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertEqual(1, photo.raw_count)
            self.assertTrue(Path(photo.preview_absolute_path).is_file())
            first_cached_preview = Path(photo.preview_absolute_path)
            BaselineAnalyzer(database).analyze_project(project.id)
            analyzed = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertNotEqual("preview-unavailable", analyzed.analysis_status)

            raw.write_bytes(b"NEW-RAW" + _jpeg(1000, 700, 170) + b"END")
            database.scan_project(project.id)
            changed = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertNotEqual(first_cached_preview, Path(changed.preview_absolute_path))
            self.assertFalse(first_cached_preview.exists())

    def test_raw_without_preview_is_review_not_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            (shoot / "unpreviewable.arw").write_bytes(b"RAW without embedded image")
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)

            BaselineAnalyzer(database).analyze_project(project.id)
            photo = database.list_photos(project.id, PhotoView.ALL)[0]

            self.assertEqual(Decision.REVIEW, photo.decision)
            self.assertEqual("preview-unavailable", photo.analysis_status)
            self.assertIn("indexed safely", photo.auto_reasons[0])

    def test_closing_project_clears_raw_preview_but_preserves_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            raw_bytes = b"RAW-HEADER" + _jpeg(1400, 900, 140) + b"RAW-END"
            raw = shoot / "frame.nef"
            raw.write_bytes(raw_bytes)
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)
            photo = database.list_photos(project.id, PhotoView.ALL)[0]
            preview = Path(photo.preview_absolute_path)
            preview_size = preview.stat().st_size
            database.set_decision([photo.id], Decision.KEEP)
            database.toggle_best([photo.id])

            removed_count, removed_bytes = database.close_project(project.id)

            self.assertEqual((1, preview_size), (removed_count, removed_bytes))
            self.assertFalse(preview.exists())
            self.assertEqual(raw_bytes, raw.read_bytes())
            self.assertEqual([], database.list_projects())
            preserved = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertIsNone(preserved.preview_absolute_path)
            self.assertEqual(Decision.KEEP, preserved.decision)
            self.assertTrue(preserved.is_best)

            reopened = database.add_project(shoot)
            database.scan_project(reopened.id)
            restored = database.list_photos(reopened.id, PhotoView.ALL)[0]
            self.assertTrue(Path(restored.preview_absolute_path).is_file())
            self.assertEqual(Decision.KEEP, restored.decision)
            self.assertTrue(restored.is_best)

    def test_representative_change_preserves_manual_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            raw_path = shoot / "frame.cr3"
            raw_path.write_bytes(b"RAW without preview")
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)
            raw_photo = database.list_photos(project.id, PhotoView.ALL)[0]
            database.set_decision([raw_photo.id], Decision.KEEP)

            (shoot / "frame.jpg").write_bytes(_jpeg())
            database.scan_project(project.id)
            paired = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertEqual(raw_photo.id, paired.id)
            self.assertEqual("frame.cr3", paired.relative_path)
            raw_path.unlink()
            database.scan_project(project.id)
            jpeg_only = database.list_photos(project.id, PhotoView.ALL)[0]

            self.assertEqual(raw_photo.id, jpeg_only.id)
            self.assertEqual("frame.jpg", jpeg_only.relative_path)
            self.assertEqual(Decision.KEEP, jpeg_only.decision)
            self.assertEqual("manual", jpeg_only.decision_source)


if __name__ == "__main__":
    unittest.main()
