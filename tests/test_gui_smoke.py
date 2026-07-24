from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel, Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QFrame, QListView, QPushButton  # noqa: E402

from photochoose.analysis_profiles import ANALYSIS_VERSION  # noqa: E402
from photochoose.app import create_application  # noqa: E402
from photochoose.fullscreen_viewer import FullscreenPhotoDialog  # noqa: E402
from photochoose.models import Decision, PhotoView  # noqa: E402


class GuiSmokeTests(unittest.TestCase):
    def test_classification_preserves_grid_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-test"], root / "state.sqlite3")
            window = app.main_window
            shoot = root / "Portraits"
            shoot.mkdir()
            for index in range(18):
                image = np.full(
                    (120, 160, 3),
                    (40 + index, 80 + index, 120 + index),
                    dtype=np.uint8,
                )
                success, encoded = cv2.imencode(".bmp", image)
                self.assertTrue(success)
                (shoot / "portrait-{0:02d}.bmp".format(index)).write_bytes(
                    encoded.tobytes()
                )

            project = window.database.add_project(shoot)
            window.database.scan_project(project.id)
            window._load_projects(project.id)
            window.resize(1080, 700)
            app.processEvents()

            selection = window.photo_grid.selectionModel()

            def selected_ids():
                return set(window._selected_photo_ids())

            single_index = window.photo_model.index(10, 0)
            single_id = single_index.data(window.photo_model.PhotoIdRole)
            selection.select(single_index, QItemSelectionModel.ClearAndSelect)
            selection.setCurrentIndex(single_index, QItemSelectionModel.NoUpdate)
            window.photo_grid.scrollTo(single_index, QListView.PositionAtCenter)
            window.photo_grid.setFocus(Qt.OtherFocusReason)
            app.processEvents()
            scroll_before = window.photo_grid.verticalScrollBar().value()
            self.assertGreater(window.photo_grid.verticalScrollBar().maximum(), 0)

            window._set_decision(Decision.REVIEW)
            app.processEvents()
            self.assertEqual({single_id}, selected_ids())
            self.assertEqual(
                single_id,
                selection.currentIndex().data(window.photo_model.PhotoIdRole),
            )
            self.assertTrue(window.photo_grid.hasFocus())
            self.assertEqual(scroll_before, window.photo_grid.verticalScrollBar().value())
            self.assertIn("portrait-10.bmp", window.inspector.details_label.text())

            multi_indexes = [window.photo_model.index(row, 0) for row in (7, 8)]
            multi_ids = {
                index.data(window.photo_model.PhotoIdRole) for index in multi_indexes
            }
            selection.select(multi_indexes[0], QItemSelectionModel.ClearAndSelect)
            selection.select(multi_indexes[1], QItemSelectionModel.Select)
            selection.setCurrentIndex(multi_indexes[1], QItemSelectionModel.NoUpdate)
            window._set_decision(Decision.KEEP)
            app.processEvents()
            self.assertEqual(multi_ids, selected_ids())
            self.assertEqual(
                multi_indexes[1].data(window.photo_model.PhotoIdRole),
                selection.currentIndex().data(window.photo_model.PhotoIdRole),
            )

            review_ids = [
                photo.id
                for photo in window.database.list_photos(project.id, PhotoView.ALL)[:4]
            ]
            window.database.set_decision(review_ids, Decision.REVIEW)
            window.view_tabs.setCurrentIndex(3)
            app.processEvents()
            self.assertGreaterEqual(window.photo_model.rowCount(), 4)
            filtered_ids = [
                window.photo_model.index(row, 0).data(window.photo_model.PhotoIdRole)
                for row in range(window.photo_model.rowCount())
            ]
            removed_index = window.photo_model.index(1, 0)
            removed_id = removed_index.data(window.photo_model.PhotoIdRole)
            expected_fallback_id = filtered_ids[2]
            selection.select(removed_index, QItemSelectionModel.ClearAndSelect)
            selection.setCurrentIndex(removed_index, QItemSelectionModel.NoUpdate)

            window._set_decision(Decision.KEEP)
            app.processEvents()
            self.assertNotIn(removed_id, selected_ids())
            self.assertEqual({expected_fallback_id}, selected_ids())
            self.assertEqual(1, selection.currentIndex().row())
            self.assertTrue(window.photo_grid.hasFocus())
            window.close()

    def test_window_loads_a_project_and_persists_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-test"], root / "state.sqlite3")
            window = app.main_window
            self.assertIn("workspaceSurface", app.styleSheet())
            library = window.findChild(QFrame, "libraryRail")
            self.assertIsNotNone(library)
            self.assertEqual(1, len(library.findChildren(QPushButton)))
            self.assertEqual(
                ["File", "Project", "Photo", "View"],
                [action.text() for action in window.menuBar().actions()],
            )
            shoot = root / "Wedding"
            shoot.mkdir()
            for index, color in enumerate(("#b14c45", "#4d7fb8", "#678f5a")):
                rgb = QColor(color)
                image = np.full(
                    (600, 800, 3),
                    (rgb.blue(), rgb.green(), rgb.red()),
                    dtype=np.uint8,
                )
                success, encoded = cv2.imencode(".bmp", image)
                self.assertTrue(success)
                (shoot / "frame-{0}.bmp".format(index)).write_bytes(encoded.tobytes())

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
            self.assertEqual("safe", window.current_project.analysis_profile)
            self.assertEqual("Safe", window.analysis_profile_combo.currentText())
            self.assertEqual(1, window.best_count_spin.value())
            window.best_count_spin.setValue(2)
            app.processEvents()
            self.assertEqual(2, window.current_project.best_count_per_group)

            balanced_index = window.analysis_profile_combo.findData("balanced")
            window.analysis_profile_combo.setCurrentIndex(balanced_index)
            app.processEvents()
            self.assertEqual("balanced", window.current_project.analysis_profile)
            self.assertTrue(
                all(
                    photo.analysis_status == "stale"
                    for photo in window.database.list_photos(project.id, PhotoView.ALL)
                )
            )
            self.assertIn("need reanalysis", window.analysis_profile_status.text())

            selection = window.photo_grid.selectionModel()
            selection.select(
                window.photo_model.index(0, 0),
                QItemSelectionModel.ClearAndSelect,
            )
            app.processEvents()
            self.assertFalse(window.inspector.preview_label.isHidden())
            self.assertFalse(window.inspector.preview_label.pixmap().isNull())
            self.assertEqual(Qt.CustomContextMenu, window.project_list.contextMenuPolicy())
            self.assertEqual(Qt.CustomContextMenu, window.photo_grid.contextMenuPolicy())
            self.assertIn("frame-0.bmp", window.inspector.details_label.text())
            self.assertIn(
                "Profile: Safe / {0}".format(ANALYSIS_VERSION),
                window.inspector.details_label.text(),
            )
            self.assertIn(
                "Model: yunet-openvino-cpu+face-mesh-openvino-cpu",
                window.inspector.details_label.text(),
            )
            self.assertIn(
                "Object model: nanodet-openvino-cpu",
                window.inspector.details_label.text(),
            )
            self.assertIn(
                "Analyze and separate",
                [action.text() for action in window._project_context_menu().actions()],
            )
            self.assertIn(
                "Calibration report…",
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
            window._open_calibration_report()
            app.processEvents()
            self.assertIsNotNone(window.calibration_dialog)
            self.assertIn(
                "Private calibration report",
                window.calibration_dialog.report_text.toPlainText(),
            )
            self.assertIn(
                "target machine",
                window.calibration_dialog.report_text.toPlainText(),
            )
            window.calibration_dialog.reject()
            app.processEvents()
            self.assertIsNone(window.calibration_dialog)
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
            self.assertEqual(
                0,
                viewer.carousel_model.photo_row(viewer.carousel.currentIndex()),
            )
            self.assertEqual("Current photo  1 of 3", viewer.carousel_position_label.text())
            self.assertLessEqual(
                abs(
                    viewer.carousel.visualRect(viewer.carousel.currentIndex()).center().x()
                    - viewer.carousel.viewport().rect().center().x()
                ),
                3,
            )
            self.assertLessEqual(
                viewer.image_label.pixmap().height(), viewer.image_label.height()
            )
            self.assertLessEqual(
                viewer.image_label.pixmap().width(), viewer.image_label.width()
            )
            self.assertEqual(0, viewer.current_index)
            viewer.next_photo()
            self.assertEqual(1, viewer.current_index)
            self.assertEqual(
                1,
                viewer.carousel_model.photo_row(viewer.carousel.currentIndex()),
            )
            viewer.next_button.setFocus()
            QTest.keyClick(viewer.next_button, Qt.Key_Left)
            self.assertEqual(0, viewer.current_index)
            viewer.previous_button.setFocus()
            QTest.keyClick(viewer.previous_button, Qt.Key_Right)
            self.assertEqual(1, viewer.current_index)
            QTest.keyClick(viewer, Qt.Key_A)
            QTest.keyClick(viewer, Qt.Key_D)
            self.assertEqual(1, viewer.current_index)
            QTest.keyClick(viewer, Qt.Key_Left)
            self.assertEqual(0, viewer.current_index)
            QTest.keyClick(viewer, Qt.Key_R)
            refreshed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertEqual(Decision.REVIEW, refreshed[0].decision)
            self.assertTrue(viewer.decision_badge.text().startswith("Review"))
            self.assertIn(
                "Close fullscreen",
                [action.text() for action in viewer._context_menu().actions()],
            )
            self.assertIn(
                "Analysis overlay",
                [action.text() for action in viewer._context_menu().actions()],
            )
            QTest.keyClick(viewer, Qt.Key_O)
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
