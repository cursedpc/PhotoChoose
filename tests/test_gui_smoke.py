from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from photochoose.app import create_application  # noqa: E402
from photochoose.fullscreen_viewer import FullscreenPhotoDialog  # noqa: E402
from photochoose.models import Decision, PhotoView  # noqa: E402


class GuiSmokeTests(unittest.TestCase):
    def test_window_loads_a_project_and_persists_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-test"], root / "state.sqlite3")
            window = app.main_window
            shoot = root / "Wedding"
            shoot.mkdir()
            for index, color in enumerate(("#b14c45", "#4d7fb8", "#678f5a")):
                image = QImage(800, 600, QImage.Format_RGB32)
                image.fill(QColor(color))
                self.assertTrue(image.save(str(shoot / "frame-{0}.jpg".format(index))))

            project = window.database.add_project(shoot)
            window._load_projects(project.id)
            window._start_scan(project, analyze_after=True)
            deadline = time.monotonic() + 10.0
            while window.busy_projects and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)

            self.assertFalse(window.busy_projects, "automatic scan and analysis timed out")
            self.assertEqual(1, window.project_list.count())
            self.assertEqual(3, window.photo_model.rowCount())
            analyzed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertTrue(all(photo.analysis_status == "complete" for photo in analyzed))

            selection = window.photo_grid.selectionModel()
            selection.select(
                window.photo_model.index(0, 0),
                QItemSelectionModel.ClearAndSelect,
            )
            self.assertEqual(Qt.CustomContextMenu, window.project_list.contextMenuPolicy())
            self.assertEqual(Qt.CustomContextMenu, window.photo_grid.contextMenuPolicy())
            self.assertIn("frame-0.jpg", window.inspector.details_label.text())
            self.assertIn(
                "Model: yunet+face-mesh-litert",
                window.inspector.details_label.text(),
            )
            self.assertIn(
                "Analyze and separate",
                [action.text() for action in window._project_context_menu().actions()],
            )
            self.assertIn(
                "Toggle Best",
                [action.text() for action in window._photo_context_menu().actions()],
            )
            window.inspector.analysis_overlay_button.setChecked(True)
            self.assertTrue(window.photo_grid.property("analysisOverlayEnabled"))
            window.inspector.toggle_button.setChecked(False)
            self.assertTrue(window.inspector.contents.isHidden())
            window.inspector.toggle_button.setChecked(True)
            self.assertFalse(window.inspector.contents.isHidden())
            window._set_decision(Decision.KEEP)
            selection.select(
                window.photo_model.index(0, 0),
                QItemSelectionModel.ClearAndSelect,
            )
            window._toggle_best()

            counts = window.database.counts(project.id)
            self.assertEqual(3, counts[PhotoView.ALL.value])
            self.assertEqual(1, counts[PhotoView.KEEP.value])
            self.assertGreaterEqual(counts[PhotoView.BEST.value], 1)
            first_photo = window.database.list_photos(project.id, PhotoView.ALL)[0]
            self.assertTrue(first_photo.is_best)
            self.assertEqual("manual", first_photo.best_source)

            selected_index = window.photo_model.index(0, 0)
            selection.select(selected_index, QItemSelectionModel.ClearAndSelect)
            deadline = time.monotonic() + 5.0
            selected_id = selected_index.data(window.photo_model.PhotoIdRole)
            while selected_id not in window.photo_model._cache and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            app.processEvents()
            viewport = window.photo_grid.viewport().grab().toImage()
            sample_point = window.photo_grid.visualRect(selected_index).center()
            selected_color = QColor(viewport.pixel(sample_point))
            expected_color = QColor("#b14c45")
            self.assertLess(abs(selected_color.red() - expected_color.red()), 20)
            self.assertLess(abs(selected_color.green() - expected_color.green()), 20)
            self.assertLess(abs(selected_color.blue() - expected_color.blue()), 20)

            viewer_photos = window.database.list_photos(project.id, PhotoView.ALL)
            viewer = FullscreenPhotoDialog(window.database, viewer_photos, 0, window)
            self.assertIsNotNone(viewer.image_label.pixmap())
            self.assertFalse(viewer.image_label.pixmap().isNull())
            viewer.resize(500, 400)
            viewer.show()
            app.processEvents()
            self.assertLessEqual(
                viewer.image_label.pixmap().height(), viewer.image_label.height()
            )
            self.assertLessEqual(
                viewer.image_label.pixmap().width(), viewer.image_label.width()
            )
            self.assertEqual(0, viewer.current_index)
            viewer.next_photo()
            self.assertEqual(1, viewer.current_index)
            QTest.keyClick(viewer, Qt.Key_Left)
            self.assertEqual(0, viewer.current_index)
            QTest.keyClick(viewer, Qt.Key_R)
            refreshed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertEqual(Decision.REVIEW, refreshed[0].decision)
            self.assertIn(
                "Close fullscreen",
                [action.text() for action in viewer._context_menu().actions()],
            )
            self.assertIn(
                "Analysis overlay",
                [action.text() for action in viewer._context_menu().actions()],
            )
            viewer.overlay_button.click()
            self.assertTrue(viewer.overlay_enabled)
            viewer.close()

            comparison_index = window.photo_model.index(0, 0)
            selection.select(comparison_index, QItemSelectionModel.ClearAndSelect)
            self.assertTrue(window.inspector.compare_button.isEnabled())
            window._open_selected_comparison()
            app.processEvents()
            self.assertIsNotNone(window.comparison_dialog)
            self.assertIsNone(window.comparison_dialog.parent())
            self.assertNotEqual(
                window.comparison_dialog.left_pane.photo.id,
                window.comparison_dialog.right_pane.photo.id,
            )
            window.comparison_dialog.zoom_slider.setValue(150)
            QTest.keyClick(window.comparison_dialog, Qt.Key_O)
            self.assertTrue(window.comparison_dialog.left_pane.overlay_enabled)
            app.processEvents()
            self.assertFalse(
                window.comparison_dialog.left_pane.image_label.pixmap().isNull()
            )
            compared_photo_id = window.comparison_dialog.left_pane.photo.id
            window.comparison_dialog._set_decision(compared_photo_id, Decision.KEEP)
            updated = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertEqual(
                Decision.KEEP,
                next(photo for photo in updated if photo.id == compared_photo_id).decision,
            )
            left_scroll = (
                window.comparison_dialog.left_pane.scroll_area.horizontalScrollBar()
            )
            right_scroll = (
                window.comparison_dialog.right_pane.scroll_area.horizontalScrollBar()
            )
            if left_scroll.maximum() and right_scroll.maximum():
                left_scroll.setValue(left_scroll.maximum())
                app.processEvents()
                self.assertEqual(right_scroll.maximum(), right_scroll.value())
            window.comparison_dialog.close()
            app.processEvents()
            self.assertIsNone(window.comparison_dialog)

            fullscreen_index = window.photo_model.index(0, 0)
            window._open_fullscreen(fullscreen_index)
            app.processEvents()
            self.assertIsNotNone(window.fullscreen_dialog)
            self.assertIsNone(window.fullscreen_dialog.parent())
            self.assertTrue(window.fullscreen_dialog.isFullScreen())
            window.fullscreen_dialog.close()
            app.processEvents()
            self.assertIsNone(window.fullscreen_dialog)

            window._close_current_project()
            self.assertEqual(0, window.project_list.count())
            self.assertEqual(0, window.photo_model.rowCount())
            reopened = window.database.add_project(shoot)
            self.assertEqual(project.id, reopened.id)
            window.close()


if __name__ == "__main__":
    unittest.main()
