from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QSize

from photochoose.analyzer import BaselineAnalyzer
from photochoose.database import PhotoDatabase
from photochoose.exporter import ProjectExporter
from photochoose.image_loader import decode_analysis_bgr, read_qimage
from photochoose.models import Decision, PhotoView

FIXTURES = Path(__file__).parent / "fixtures"


class ImageFormatTests(unittest.TestCase):
    def test_heic_fixture_decodes_for_analysis_and_qt_views(self) -> None:
        fixture = FIXTURES / "sample.heic"

        bgr = decode_analysis_bgr(fixture)
        preview = read_qimage(fixture, QSize(48, 48))

        self.assertEqual((96, 96, 3), bgr.shape)
        self.assertFalse(preview.isNull())
        self.assertLessEqual(preview.width(), 48)
        self.assertLessEqual(preview.height(), 48)

    def test_transparent_png_is_composited_consistently_for_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "alpha.png"
            image = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
            image.putpixel((1, 0), (20, 80, 200, 255))
            image.save(path)

            bgr = decode_analysis_bgr(path)
            preview = read_qimage(path)

            np.testing.assert_array_equal(bgr[0, 0], np.array([127, 127, 127]))
            np.testing.assert_array_equal(bgr[0, 1], np.array([200, 80, 20]))
            self.assertFalse(preview.isNull())

    def test_live_photo_companions_are_one_item_and_export_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            shutil.copyfile(FIXTURES / "sample.heic", shoot / "IMG_0001.HEIC")
            (shoot / "IMG_0001.MOV").write_bytes(b"motion companion")
            (shoot / "IMG_0001.AAE").write_text(
                "<plist><dict><key>assetIdentifier</key>"
                "<string>live-0001</string></dict></plist>",
                encoding="utf-8",
            )
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)

            self.assertEqual(1, database.scan_project(project.id))
            photo = database.list_photos(project.id, PhotoView.ALL)[0]

            self.assertEqual("IMG_0001.HEIC", photo.relative_path)
            self.assertEqual(
                ("IMG_0001.HEIC", "IMG_0001.MOV", "IMG_0001.AAE"),
                photo.asset_paths,
            )
            self.assertEqual(1, photo.motion_count)
            self.assertEqual(1, photo.sidecar_count)
            self.assertEqual("live-0001", photo.live_photo_id)
            self.assertEqual("LIVE", photo.asset_badge)

            database.set_decision([photo.id], Decision.KEEP)
            destination = root / "Export"
            result = ProjectExporter(database).export_keep(
                project.id,
                destination,
            )

            self.assertEqual(3, result.copied)
            self.assertEqual(b"motion companion", (destination / "IMG_0001.MOV").read_bytes())
            self.assertTrue((destination / "IMG_0001.HEIC").is_file())
            self.assertTrue((destination / "IMG_0001.AAE").is_file())

    def test_heic_analysis_completes_with_shared_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            shutil.copyfile(FIXTURES / "sample.heic", shoot / "frame.heic")
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)

            BaselineAnalyzer(database).analyze_project(project.id)
            photo = database.list_photos(project.id, PhotoView.ALL)[0]

            self.assertEqual("complete", photo.analysis_status)
            self.assertIsNotNone(photo.focus_score)
            self.assertIsNotNone(photo.perceptual_hash)

    def test_explicit_burst_identifier_groups_separate_heic_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shoot = root / "Shoot"
            shoot.mkdir()
            for index, value in enumerate((45, 205), start=1):
                image = Image.new("RGB", (96, 96), (value, value, value))
                image.save(shoot / "burst-{0}.png".format(index))
                (shoot / "burst-{0}.xmp".format(index)).write_text(
                    '<x:xmpmeta burstUUID="burst-shared-id" />',
                    encoding="utf-8",
                )
            database = PhotoDatabase(root / "state.sqlite3")
            project = database.add_project(shoot)
            database.scan_project(project.id)

            BaselineAnalyzer(database).analyze_project(project.id)
            photos = database.list_photos(project.id, PhotoView.ALL)

            self.assertEqual({"burst-shared-id"}, {photo.burst_id for photo in photos})
            self.assertEqual(1, len({photo.similarity_group for photo in photos}))
            self.assertTrue(photos[0].similarity_group.startswith("B"))
            self.assertEqual("Burst metadata · 2 frames", photos[0].group_reason)


if __name__ == "__main__":
    unittest.main()
