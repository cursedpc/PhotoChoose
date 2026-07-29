from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402

from photochoose.app import create_application  # noqa: E402
from photochoose.database import PhotoDatabase  # noqa: E402
from photochoose.models import PhotoView  # noqa: E402
from photochoose.thumbnail_model import PhotoListModel  # noqa: E402


class ThumbnailModelTests(unittest.TestCase):
    def test_rescan_rejects_cached_and_late_thumbnails_for_replaced_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(
                ["photochoose-thumbnail-generation-test"], root / "app.sqlite3"
            )
            database = PhotoDatabase(root / "state.sqlite3")
            shoot = root / "Shoot"
            shoot.mkdir()
            first_path = shoot / "first.jpg"
            first_path.write_bytes(b"first image placeholder")
            project = database.add_project(shoot)
            database.scan_project(project.id)
            first = database.list_photos(project.id, PhotoView.ALL)[0]

            model = PhotoListModel(database)
            model.set_photos([first])
            old_generation = model._thumbnail_generation
            stale_image = QImage(16, 16, QImage.Format_RGB32)
            stale_image.fill(QColor("#ff0000"))
            model._thumbnail_loaded(first.id, old_generation, stale_image)
            self.assertIn(first.id, model._cache)
            model.set_photos([first])
            self.assertIn(first.id, model._cache)

            first_path.write_bytes(b"replacement image contents")
            database.scan_project(project.id)
            replacement = database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertEqual(first.id, replacement.id)

            model.set_photos([replacement], invalidate_thumbnails=True)
            self.assertEqual({}, model._cache)
            model._thumbnail_loaded(first.id, old_generation, stale_image)
            self.assertEqual({}, model._cache)

            fresh_image = QImage(16, 16, QImage.Format_RGB32)
            fresh_image.fill(QColor("#0000ff"))
            model._thumbnail_loaded(
                replacement.id,
                model._thumbnail_generation,
                fresh_image,
            )
            self.assertEqual(
                QColor("#0000ff"),
                model._cache[replacement.id].toImage().pixelColor(0, 0),
            )
            app.main_window.close()


if __name__ == "__main__":
    unittest.main()
