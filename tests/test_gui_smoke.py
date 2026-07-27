from __future__ import annotations

import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import (  # noqa: E402
    QEvent,
    QItemSelectionModel,
    QMimeData,
    QPoint,
    QPointF,
    Qt,
    QUrl,
)
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QDragEnterEvent,
    QDragLeaveEvent,
    QKeyEvent,
    QWheelEvent,
)
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QFrame, QListView, QPushButton  # noqa: E402

from photochoose.analysis_profiles import ANALYSIS_VERSION  # noqa: E402
from photochoose.app import create_application  # noqa: E402
from photochoose.fullscreen_viewer import FullscreenPhotoDialog  # noqa: E402
from photochoose.main_window import _progress_message  # noqa: E402
from photochoose.models import (  # noqa: E402
    Decision,
    EyeEvidence,
    FaceEvidence,
    ObjectEvidence,
    PhotoView,
)
from photochoose.selection_inspector import SelectionInspector  # noqa: E402


class GuiSmokeTests(unittest.TestCase):
    def test_inspector_scrolls_all_controls_on_short_displays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(
                ["photochoose-inspector-scroll-test"], root / "state.sqlite3"
            )
            inspector = SelectionInspector()
            inspector.resize(310, 340)
            inspector.toggle_button.setChecked(True)
            inspector.show()
            app.processEvents()

            scroll_bar = inspector.panel_scroll.verticalScrollBar()
            self.assertGreater(scroll_bar.maximum(), 0)
            scroll_bar.setValue(scroll_bar.maximum())
            app.processEvents()
            self.assertEqual(scroll_bar.maximum(), scroll_bar.value())
            self.assertFalse(inspector.contents.isHidden())

            inspector.close()
            app.main_window.close()

    def test_dropped_folders_are_filtered_and_processed_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = create_application(["photochoose-drop-test"], root / "state.sqlite3")
            window = app.main_window
            self.assertEqual(
                "clsid:0AC0837C-BBF8-452A-850D-79D08E667CA7",
                window._photo_source_start_url("win32").toString(),
            )
            self.assertTrue(window.acceptDrops())
            self.assertFalse(window.photo_grid.acceptDrops())
            first = root / "First shoot"
            second = root / "Second shoot"
            first.mkdir()
            second.mkdir()
            ordinary_file = root / "photo.jpg"
            ordinary_file.write_bytes(b"not needed for this import test")
            mime_data = QMimeData()
            mime_data.setUrls(
                [
                    QUrl.fromLocalFile(str(first)),
                    QUrl.fromLocalFile(str(ordinary_file)),
                    QUrl("https://example.com/not-local"),
                    QUrl.fromLocalFile(str(first)),
                    QUrl.fromLocalFile(str(second)),
                ]
            )

            folders = window._folders_from_mime_data(mime_data)
            self.assertEqual([first.resolve(), second.resolve()], folders)
            drag_enter = QDragEnterEvent(
                QPoint(10, 10),
                Qt.CopyAction,
                mime_data,
                Qt.LeftButton,
                Qt.NoModifier,
            )
            window.dragEnterEvent(drag_enter)
            self.assertTrue(drag_enter.isAccepted())
            self.assertTrue(window.project_list.property("dropActive"))
            drag_leave = QDragLeaveEvent()
            window.dragLeaveEvent(drag_leave)
            self.assertFalse(window.project_list.property("dropActive"))

            started = []
            window._start_scan = lambda project, analyze_after=False: started.append(
                (project, analyze_after)
            )
            window._queue_project_folders(folders)
            self.assertEqual(1, len(started))
            self.assertEqual(first.resolve(), Path(started[0][0].root_path))
            self.assertTrue(started[0][1])
            self.assertEqual(1, len(window._project_import_queue))

            window._complete_import_project(started[0][0].id)
            app.processEvents()
            self.assertEqual(2, len(started))
            self.assertEqual(second.resolve(), Path(started[1][0].root_path))
            self.assertTrue(started[1][1])
            window._complete_import_project(started[1][0].id)
            self.assertIsNone(window._active_import_project_id)
            self.assertEqual([], window._project_import_queue)

    def test_progress_message_estimates_remaining_time(self) -> None:
        self.assertEqual(
            "Scanning… 0 / 20 · estimating time…",
            _progress_message("Scanning…", 0, 20, 0.5),
        )
        self.assertEqual(
            "Analyzing locally… 10 / 30 · ETA 4 min",
            _progress_message("Analyzing locally…", 10, 30, 120.0),
        )
        self.assertEqual(
            "Analyzing locally… 29 / 30 · ETA less than 1 min",
            _progress_message("Analyzing locally…", 29, 30, 30.0),
        )

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
            self.assertFalse(app.windowIcon().isNull())
            library = window.findChild(QFrame, "libraryRail")
            self.assertIsNotNone(library)
            self.assertEqual(1, len(library.findChildren(QPushButton)))
            self.assertEqual(
                ["File", "Folder", "Photo", "View", "Help"],
                [action.text() for action in window.menuBar().actions()],
            )
            self.assertEqual("Show hints", window.show_hints_action.text())
            self.assertTrue(window.folder_menu.menuAction().isEnabled())
            self.assertTrue(window.photo_menu.menuAction().isEnabled())
            self.assertTrue(window.view_menu.menuAction().isEnabled())
            self.assertTrue(
                all(
                    not action.isEnabled()
                    for action in window.folder_menu.actions()
                    if not action.isSeparator()
                )
            )
            self.assertTrue(
                all(not action.isEnabled() for action in window.photo_menu_actions)
            )
            self.assertTrue(
                all(not action.isEnabled() for action in window.view_menu_actions)
            )
            self.assertFalse(window.view_tabs.isEnabled())
            self.assertFalse(window.photo_grid.isEnabled())
            self.assertEqual("No folder selected", window.project_title.text())
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
            self.assertTrue(
                all(action.isEnabled() for action in window.photo_menu_actions)
            )
            self.assertTrue(
                all(action.isEnabled() for action in window.view_menu_actions)
            )
            window._start_scan(project, analyze_after=True)
            deadline = time.monotonic() + 10.0
            while window.busy_projects and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)

            self.assertFalse(window.busy_projects, "automatic scan and analysis timed out")
            self.assertEqual(1, window.project_list.count())
            self.assertEqual(3, window.photo_model.rowCount())
            app.processEvents()
            required_tab_width = sum(
                window.view_tabs.tabRect(index).width()
                for index in range(window.view_tabs.count())
            )
            self.assertGreaterEqual(window.view_tabs.width(), required_tab_width)
            for index in range(1, window.view_tabs.count()):
                self.assertGreaterEqual(
                    window.view_tabs.tabRect(index).left(),
                    window.view_tabs.tabRect(index - 1).right(),
                )
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
                "Reanalyze",
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
            displayed_size = viewer.image_label.pixmap().deviceIndependentSize()
            self.assertLessEqual(
                displayed_size.height(), viewer.image_scroll.viewport().height()
            )
            self.assertLessEqual(
                displayed_size.width(), viewer.image_scroll.viewport().width()
            )
            self.assertAlmostEqual(
                viewer._current_device_pixel_ratio,
                viewer.image_label.pixmap().devicePixelRatio(),
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
            fullscreen_pixmap_key = viewer.image_label.pixmap().cacheKey()
            viewer.zoom_slider.setValue(200)
            app.processEvents()
            self.assertEqual(200, viewer.zoom_percent)
            self.assertEqual(
                fullscreen_pixmap_key, viewer.image_label.pixmap().cacheKey()
            )
            self.assertTrue(
                viewer.image_scroll.horizontalScrollBar().maximum() > 0
                or viewer.image_scroll.verticalScrollBar().maximum() > 0
            )
            horizontal_scroll = viewer.image_scroll.horizontalScrollBar()
            vertical_scroll = viewer.image_scroll.verticalScrollBar()
            pan_before = (horizontal_scroll.value(), vertical_scroll.value())
            pan_start = viewer.image_scroll.viewport().rect().center()
            QTest.mousePress(
                viewer.image_scroll.viewport(), Qt.LeftButton, pos=pan_start
            )
            QTest.mouseMove(
                viewer.image_scroll.viewport(), pan_start - QPoint(24, 18), delay=10
            )
            QTest.mouseRelease(
                viewer.image_scroll.viewport(),
                Qt.LeftButton,
                pos=pan_start - QPoint(24, 18),
            )
            self.assertNotEqual(
                pan_before, (horizontal_scroll.value(), vertical_scroll.value())
            )
            zoom_before_wheel = viewer.zoom_slider.value()
            wheel = QWheelEvent(
                QPointF(10, 10),
                QPointF(10, 10),
                QPoint(),
                QPoint(0, 120),
                Qt.NoButton,
                Qt.ControlModifier,
                Qt.NoScrollPhase,
                False,
            )
            viewer.image_scroll.wheelEvent(wheel)
            QTest.qWait(20)
            app.processEvents()
            self.assertGreater(viewer.zoom_slider.value(), zoom_before_wheel)
            viewer.zoom_slider.setValue(100)
            viewer._adjust_zoom(1.004)
            self.assertEqual(100, viewer.zoom_slider.value())
            viewer._adjust_zoom(1.004)
            self.assertEqual(101, viewer.zoom_slider.value())
            viewer.zoom_slider.setValue(100)
            for key, expected_decision in (
                (Qt.Key_K, Decision.KEEP),
                (Qt.Key_R, Decision.REVIEW),
                (Qt.Key_X, Decision.REJECT),
            ):
                viewer.menu_button.setFocus()
                QTest.keyClick(viewer.menu_button, key)
                refreshed = window.database.list_photos(project.id, PhotoView.ALL)
                self.assertEqual(expected_decision, refreshed[0].decision)
                self.assertTrue(
                    viewer.decision_badge.text().startswith(
                        expected_decision.value.title()
                    )
                )
            viewer.activateWindow()
            app.processEvents()
            non_latin_keep = QKeyEvent(
                QEvent.KeyPress,
                ord("Л"),
                Qt.NoModifier,
                0x25,
                0,
                0,
                "л",
            )
            with patch("photochoose.keyboard_shortcuts.sys.platform", "win32"):
                app.sendEvent(viewer.menu_button, non_latin_keep)
            refreshed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertEqual(Decision.KEEP, refreshed[0].decision)
            non_latin_macos_reject = QKeyEvent(
                QEvent.KeyPress,
                ord("Ч"),
                Qt.NoModifier,
                0,
                0x07,
                0,
                "ч",
            )
            with patch("photochoose.keyboard_shortcuts.sys.platform", "darwin"):
                app.sendEvent(viewer.menu_button, non_latin_macos_reject)
            refreshed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertEqual(Decision.REJECT, refreshed[0].decision)
            viewer.next_button.setFocus()
            was_best = refreshed[0].is_best
            QTest.keyClick(viewer.next_button, Qt.Key_B)
            refreshed = window.database.list_photos(project.id, PhotoView.ALL)
            self.assertNotEqual(was_best, refreshed[0].is_best)
            self.assertIn(
                "Close fullscreen",
                [action.text() for action in viewer._context_menu().actions()],
            )
            self.assertIn(
                "Analysis overlay",
                [action.text() for action in viewer._context_menu().actions()],
            )
            overlay_eye = EyeEvidence(
                state="open",
                aperture=0.2,
                openness_score=0.9,
                focus_score=100.0,
                landmarks=((0.3, 0.3), (0.34, 0.3), (0.32, 0.33)),
            )
            viewer.photos[0] = replace(
                viewer.photos[0],
                subject_regions=(
                    FaceEvidence(
                        (0.2, 0.2, 0.3, 0.4),
                        0.95,
                        120.0,
                        True,
                        (overlay_eye,),
                    ),
                ),
                subject_objects=(
                    ObjectEvidence(
                        "person", (0.2, 0.2, 0.3, 0.4), 0.9, 120.0, "primary"
                    ),
                ),
            )
            viewer._show_current_photo()
            overlay_base_pixmap_key = viewer.image_label.pixmap().cacheKey()
            viewer.previous_button.setFocus()
            QTest.keyClick(viewer.previous_button, Qt.Key_O)
            self.assertTrue(viewer.overlay_enabled)
            self.assertEqual(
                overlay_base_pixmap_key, viewer.image_label.pixmap().cacheKey()
            )
            self.assertTrue(viewer.overlay_controls_scroll.isVisible())
            self.assertEqual(
                ["Common", "All", "Hide"],
                [
                    viewer.overlay_global_combo.itemText(index)
                    for index in range(3)
                ],
            )
            QTest.mouseClick(
                viewer.overlay_global_combo,
                Qt.LeftButton,
                pos=viewer.overlay_global_combo.rect().center(),
            )
            app.processEvents()
            self.assertTrue(viewer.overlay_global_combo.view().isVisible())
            QTest.keyClick(viewer.overlay_global_combo, Qt.Key_Escape)
            self.assertEqual(
                ["Common", "All", "Hide"],
                [
                    viewer.overlay_category_combos["faces"].itemText(index)
                    for index in range(3)
                ],
            )
            faces_mode_combo = viewer.overlay_category_combos["faces"]
            expected_option_width = max(
                faces_mode_combo.fontMetrics().horizontalAdvance(
                    faces_mode_combo.itemText(index)
                )
                for index in range(faces_mode_combo.count())
            )
            self.assertEqual(expected_option_width + 46, faces_mode_combo.width())
            self.assertEqual(
                max(20, faces_mode_combo.fontMetrics().height() + 4),
                faces_mode_combo.height(),
            )
            QTest.mouseClick(
                faces_mode_combo,
                Qt.LeftButton,
                pos=faces_mode_combo.rect().center(),
            )
            app.processEvents()
            self.assertTrue(faces_mode_combo.view().isVisible())
            QTest.keyClick(faces_mode_combo, Qt.Key_Escape)
            viewer.overlay_global_combo.setCurrentText("All")
            app.processEvents()
            all_visibility = viewer._current_overlay_visibility()
            self.assertEqual("all", all_visibility.faces.value)
            self.assertEqual("all", all_visibility.eyes.value)
            self.assertEqual("all", all_visibility.objects.value)
            self.assertTrue(
                viewer.overlay_item_toggles[("objects", 0, -1)].isChecked()
            )
            viewer.overlay_global_combo.setCurrentText("Common")
            app.processEvents()
            face_toggle = viewer.overlay_item_toggles[("faces", 0, -1)]
            eye_toggle = viewer.overlay_item_toggles[("eyes", 0, 0)]
            object_toggle = viewer.overlay_item_toggles[("objects", 0, -1)]
            self.assertTrue(face_toggle.isChecked())
            self.assertTrue(eye_toggle.isChecked())
            self.assertFalse(object_toggle.isChecked())
            QTest.mouseClick(face_toggle, Qt.LeftButton, pos=QPoint(2, 9))
            self.assertFalse(face_toggle.isChecked())
            self.assertEqual(-1, viewer.overlay_global_combo.currentIndex())
            self.assertEqual("Mixed", viewer.overlay_global_combo.placeholderText())
            QTest.mouseClick(face_toggle, Qt.LeftButton, pos=QPoint(31, 9))
            self.assertTrue(face_toggle.isChecked())
            QTest.mouseClick(face_toggle, Qt.LeftButton, pos=QPoint(2, 9))
            self.assertFalse(face_toggle.isChecked())
            viewer.overlay_category_combos["eyes"].setCurrentText("Hide")
            eye_toggle = viewer.overlay_item_toggles[("eyes", 0, 0)]
            object_toggle = viewer.overlay_item_toggles[("objects", 0, -1)]
            self.assertFalse(eye_toggle.isChecked())
            self.assertFalse(object_toggle.isChecked())
            eye_toggle.setChecked(True)
            object_toggle.setChecked(True)
            app.processEvents()
            first_visibility = viewer._current_overlay_visibility()
            self.assertEqual("hide", first_visibility.face_overrides[0][1].value)
            self.assertEqual("show", first_visibility.eye_overrides[0][2].value)
            self.assertEqual("show", first_visibility.object_overrides[0][1].value)
            viewer.next_photo()
            self.assertEqual("Common", viewer.overlay_category_combos["faces"].currentText())
            viewer.previous_photo()
            self.assertFalse(
                viewer.overlay_item_toggles[("faces", 0, -1)].isChecked()
            )
            self.assertTrue(
                viewer.overlay_item_toggles[("eyes", 0, 0)].isChecked()
            )
            viewer._set_analysis_overlay(False)
            self.assertFalse(viewer.overlay_controls_scroll.isVisible())
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
            comparison_pixmap_key = (
                window.comparison_dialog.left_pane.image_label.pixmap().cacheKey()
            )
            window.comparison_dialog.zoom_slider.setValue(150)
            self.assertEqual(
                comparison_pixmap_key,
                window.comparison_dialog.left_pane.image_label.pixmap().cacheKey(),
            )
            QTest.keyClick(window.comparison_dialog, Qt.Key_O)
            self.assertTrue(window.comparison_dialog.left_pane.overlay_enabled)
            non_latin_macos_overlay = QKeyEvent(
                QEvent.KeyPress,
                ord("Щ"),
                Qt.NoModifier,
                0,
                0x1F,
                0,
                "щ",
            )
            with patch("photochoose.keyboard_shortcuts.sys.platform", "darwin"):
                app.sendEvent(window.comparison_dialog, non_latin_macos_overlay)
            self.assertFalse(window.comparison_dialog.left_pane.overlay_enabled)
            app.processEvents()
            self.assertFalse(
                window.comparison_dialog.left_pane.image_label.pixmap().isNull()
            )
            comparison_zoom = window.comparison_dialog.zoom_slider.value()
            comparison_wheel = QWheelEvent(
                QPointF(10, 10),
                QPointF(10, 10),
                QPoint(),
                QPoint(0, 120),
                Qt.NoButton,
                Qt.ControlModifier,
                Qt.NoScrollPhase,
                False,
            )
            window.comparison_dialog.left_pane.scroll_area.wheelEvent(
                comparison_wheel
            )
            QTest.qWait(20)
            app.processEvents()
            self.assertGreater(
                window.comparison_dialog.zoom_slider.value(), comparison_zoom
            )
            app.processEvents()
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

            window.show()
            window.activateWindow()
            window.project_list.setFocus()
            app.processEvents()
            QTest.keyClick(window.project_list, Qt.Key_Delete)
            self.assertEqual(0, window.project_list.count())
            self.assertEqual(0, window.photo_model.rowCount())
            reopened = window.database.add_project(shoot)
            self.assertEqual(project.id, reopened.id)
            window.close()


if __name__ == "__main__":
    unittest.main()
